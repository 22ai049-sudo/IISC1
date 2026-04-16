"""
VisDrone MOT assignment starter (Colab-friendly)
Author: generated for a custom tracking-by-detection baseline.

How to use in Colab:
1) Upload this file and run it cell-by-cell OR copy sections into a notebook.
2) Upload VisDrone zip files to /content/sample_data/.
3) Run the configuration section.
"""

# =========================
# 0) Install dependencies
# =========================
# In Colab, uncomment and run:
# !pip -q install ultralytics motmetrics opencv-python tqdm pandas numpy scipy


# =========================
# 1) Imports
# =========================
from __future__ import annotations
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from ultralytics import YOLO


# =========================
# 2) Configuration
# =========================
# Place your zip in /content/sample_data, e.g. VisDrone2019-MOT-val.zip
COLAB_SAMPLE_DIR = Path('/content/sample_data')
WORK_DIR = Path('/content/visdrone_work')
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Choose split zip(s) to evaluate on. You can include train/val/test-dev as available.
ZIP_FILES = [
    # COLAB_SAMPLE_DIR / 'VisDrone2019-MOT-val.zip',
]

# Detector config
YOLO_WEIGHTS = 'yolov8n.pt'      # lightweight baseline
DETECT_IMGSZ = 960
DETECT_CONF = 0.20
DETECT_IOU = 0.50

# Tracker config
MIN_BOX_AREA = 80
TRACK_MAX_AGE = 18
TRACK_MIN_HITS = 2
IOU_MATCH_THRESH = 0.30

# VisDrone classes commonly used for MOT
# 1: pedestrian, 2: people, 4: car, 5: van, 6: truck, 9: bus
VISDRONE_VALID_CLASSES = {1, 2, 4, 5, 6, 9}

# Map YOLO COCO classes to pseudo-VisDrone ids for consistent filtering
# COCO: person=0, car=2, bus=5, truck=7
COCO_TO_VISDRONE = {
    0: 1,
    2: 4,
    5: 9,
    7: 6,
}


# =========================
# 3) Data helpers
# =========================
def unzip_archives(zip_paths: list[Path], out_root: Path) -> list[Path]:
    extracted = []
    for z in zip_paths:
        z = Path(z)
        if not z.exists():
            print(f'[WARN] Zip not found: {z}')
            continue
        target = out_root / z.stem
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, 'r') as f:
                f.extractall(target)
        extracted.append(target)
    return extracted


def find_sequence_roots(extracted_dirs: list[Path]) -> list[Path]:
    seq_dirs = []
    for base in extracted_dirs:
        # VisDrone layout usually has sequences under .../sequences
        cand = list(base.rglob('sequences'))
        if not cand:
            continue
        for seq_parent in cand:
            for seq in sorted([p for p in seq_parent.iterdir() if p.is_dir()]):
                seq_dirs.append(seq)
    return sorted(seq_dirs)


def find_annotation_file(sequence_dir: Path) -> Path | None:
    seq_name = sequence_dir.name
    for candidate in sequence_dir.parents:
        ann_dir = candidate / 'annotations'
        ann_file = ann_dir / f'{seq_name}.txt'
        if ann_file.exists():
            return ann_file
    return None


def load_visdrone_gt(ann_path: Path) -> dict[int, list[dict]]:
    """
    Returns frame-indexed GT dict:
      frame -> [{'id': int, 'bbox': [x1,y1,x2,y2], 'cls': int, 'occ': int, 'trunc': int}, ...]
    """
    cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'cls', 'trunc', 'occ']
    df = pd.read_csv(ann_path, header=None, names=cols)

    gt_by_frame = defaultdict(list)
    for _, r in df.iterrows():
        if int(r['cls']) not in VISDRONE_VALID_CLASSES:
            continue
        x1, y1 = float(r['x']), float(r['y'])
        x2, y2 = x1 + float(r['w']), y1 + float(r['h'])
        gt_by_frame[int(r['frame'])].append({
            'id': int(r['id']),
            'bbox': [x1, y1, x2, y2],
            'cls': int(r['cls']),
            'trunc': int(r['trunc']),
            'occ': int(r['occ']),
        })
    return gt_by_frame


# =========================
# 4) Tracking core
# =========================
def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: Nx4, b:Mx4 in xyxy"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


@dataclass
class Track:
    tid: int
    bbox: np.ndarray  # xyxy
    cls: int
    conf: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0

    def predict(self):
        # Simple constant-box tracker; intentionally lightweight.
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: np.ndarray, conf: float, cls: int):
        # Exponential smoothing on box for jitter reduction
        self.bbox = 0.7 * self.bbox + 0.3 * bbox
        self.conf = 0.6 * self.conf + 0.4 * conf
        self.cls = cls
        self.hits += 1
        self.time_since_update = 0


class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=15, min_hits=2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.next_id = 1

    def _associate(self, dets: np.ndarray):
        if len(self.tracks) == 0 or len(dets) == 0:
            return [], list(range(len(self.tracks))), list(range(len(dets)))

        trk_boxes = np.stack([t.bbox for t in self.tracks], axis=0)
        det_boxes = dets[:, :4]
        iou = bbox_iou_matrix(trk_boxes, det_boxes)
        cost = 1.0 - iou

        r, c = linear_sum_assignment(cost)
        matches = []
        unmatched_t = set(range(len(self.tracks)))
        unmatched_d = set(range(len(dets)))

        for ti, di in zip(r, c):
            if iou[ti, di] >= self.iou_thresh:
                matches.append((ti, di))
                unmatched_t.discard(ti)
                unmatched_d.discard(di)

        return matches, sorted(list(unmatched_t)), sorted(list(unmatched_d))

    def update(self, detections: np.ndarray) -> list[Track]:
        # detections: Nx6 -> x1,y1,x2,y2,conf,cls
        for t in self.tracks:
            t.predict()

        matches, unmatched_t, unmatched_d = self._associate(detections)

        for ti, di in matches:
            det = detections[di]
            self.tracks[ti].update(det[:4], float(det[4]), int(det[5]))

        for di in unmatched_d:
            d = detections[di]
            self.tracks.append(
                Track(
                    tid=self.next_id,
                    bbox=d[:4].copy(),
                    cls=int(d[5]),
                    conf=float(d[4]),
                )
            )
            self.next_id += 1

        # prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # expose confirmed and recently updated tracks
        output = [
            t for t in self.tracks
            if t.hits >= self.min_hits and t.time_since_update == 0
        ]
        return output


# =========================
# 5) Detector wrapper
# =========================
class YOLODetector:
    def __init__(self, weights='yolov8n.pt', imgsz=960, conf=0.25, iou=0.50):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            source=img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)

        rows = []
        for b, s, c in zip(xyxy, conf, cls):
            if c not in COCO_TO_VISDRONE:
                continue
            x1, y1, x2, y2 = b.astype(float)
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue
            rows.append([x1, y1, x2, y2, float(s), COCO_TO_VISDRONE[c]])

        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


# =========================
# 6) MOT evaluation
# =========================
def evaluate_sequence(
    sequence_dir: Path,
    detector: YOLODetector,
    tracker: IoUTracker,
    use_gt_ann: bool = True,
):
    ann = find_annotation_file(sequence_dir)
    if ann is None or (not ann.exists()):
        raise FileNotFoundError(f'Annotation not found for sequence: {sequence_dir.name}')

    gt_by_frame = load_visdrone_gt(ann)
    frame_paths = sorted(sequence_dir.glob('*.jpg'))
    acc = mm.MOTAccumulator(auto_id=True)

    predictions_for_export = []

    for i, fp in enumerate(tqdm(frame_paths, desc=sequence_dir.name), start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        dets = detector.detect(img)
        active_tracks = tracker.update(dets)

        gt_objs = gt_by_frame.get(i, [])
        gt_ids = [g['id'] for g in gt_objs]
        gt_boxes = np.array([g['bbox'] for g in gt_objs], dtype=np.float32) if gt_objs else np.zeros((0, 4), np.float32)

        pred_ids = [t.tid for t in active_tracks]
        pred_boxes = np.array([t.bbox for t in active_tracks], dtype=np.float32) if active_tracks else np.zeros((0, 4), np.float32)

        # MOTMetrics expects distance (lower is better), NaN means impossible match.
        iou = bbox_iou_matrix(gt_boxes, pred_boxes)
        dist = 1.0 - iou
        if dist.size:
            dist[iou < 0.5] = np.nan

        acc.update(gt_ids, pred_ids, dist)

        for t in active_tracks:
            x1, y1, x2, y2 = t.bbox.tolist()
            w, h = x2 - x1, y2 - y1
            predictions_for_export.append([
                i, t.tid, x1, y1, w, h, t.conf, t.cls, -1, -1
            ])

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr', 'num_switches', 'num_misses', 'num_false_positives'],
        name=sequence_dir.name
    )

    pred_df = pd.DataFrame(
        predictions_for_export,
        columns=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'cls', 'trunc', 'occ']
    )
    return summary, pred_df


# =========================
# 7) End-to-end runner
# =========================
def run_full_pipeline(zip_files: list[Path]):
    extracted = unzip_archives(zip_files, WORK_DIR)
    if not extracted:
        raise RuntimeError('No dataset was extracted. Check ZIP_FILES paths.')

    seqs = find_sequence_roots(extracted)
    if not seqs:
        raise RuntimeError('No sequence folders found under extracted archives.')

    print(f'Found {len(seqs)} sequences.')

    detector = YOLODetector(weights=YOLO_WEIGHTS, imgsz=DETECT_IMGSZ, conf=DETECT_CONF, iou=DETECT_IOU)

    summaries = []
    out_pred_root = WORK_DIR / 'predictions'
    out_pred_root.mkdir(exist_ok=True, parents=True)

    for seq in seqs:
        tracker = IoUTracker(
            iou_thresh=IOU_MATCH_THRESH,
            max_age=TRACK_MAX_AGE,
            min_hits=TRACK_MIN_HITS,
        )
        seq_summary, pred_df = evaluate_sequence(seq, detector, tracker)
        summaries.append(seq_summary)

        pred_path = out_pred_root / f'{seq.name}.txt'
        pred_df.to_csv(pred_path, header=False, index=False)

    all_summary = pd.concat(summaries)
    print('\nPer-sequence metrics:')
    display(all_summary)

    # Aggregate mean over numeric metrics
    numeric_cols = [c for c in all_summary.columns if np.issubdtype(all_summary[c].dtype, np.number)]
    mean_row = all_summary[numeric_cols].mean(numeric_only=True).to_frame().T
    mean_row.index = ['MEAN']
    print('\nAggregate (mean) metrics:')
    display(mean_row)

    return all_summary, mean_row, out_pred_root


# =========================
# 8) Run
# =========================
# Example usage in Colab:
# ZIP_FILES = [Path('/content/sample_data/VisDrone2019-MOT-val.zip')]
# all_summary, mean_summary, pred_dir = run_full_pipeline(ZIP_FILES)
# print('Predictions saved in:', pred_dir)


# =========================
# 9) Suggested write-up template
# =========================
WRITEUP_TEMPLATE = """
### Design choices
1. **Detector**: YOLOv8n pretrained on COCO for speed and easy deployment in Colab.
2. **Tracker**: Custom IoU-based online tracker with Hungarian matching.
3. **Filtering**: Class mapping from COCO to VisDrone MOT categories and min-area filtering.
4. **Stability**: Box-coordinate exponential smoothing to reduce jitter.

### Observed limitations
1. **Domain gap**: COCO-pretrained detector misses small/occluded drone-view objects.
2. **No motion model**: Constant-box prediction hurts during camera movement and abrupt motion.
3. **No re-ID embedding**: Identity switches increase under long occlusions/crowds.
4. **Class mismatch**: Some VisDrone categories are not cleanly represented in COCO.

### Potential improvements
1. Fine-tune YOLO on VisDrone MOT train split.
2. Replace tracker with ByteTrack/BoT-SORT + appearance embeddings.
3. Add camera-motion compensation.
4. Tune confidence/IoU thresholds per sequence type.
"""

print('Script loaded. Set ZIP_FILES, then call run_full_pipeline(ZIP_FILES).')
