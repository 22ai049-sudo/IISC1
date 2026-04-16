# VisDrone MOT Submission Guide (No-Code)

This guide is for your exact setup: you will upload the VisDrone ZIP file yourself into **Colab → sample_data**.

## 1) What to submit
- One Google Colab notebook.
- The notebook must contain:
  - end-to-end pipeline execution,
  - clearly printed MOT metrics,
  - short analysis of design choices and limitations.
- Share the notebook link with viewer access enabled.

## 2) Dataset handling expectation
- Keep dataset input path centered on `/content/sample_data`.
- Unzip once, then reuse extracted folders to avoid repeated runtime.
- Verify sequence folders and annotation files are discovered before running experiments.

## 3) Pipeline structure to present
- Detection stage (model + confidence/IoU thresholds).
- Tracking stage (association logic + track lifecycle).
- Evaluation stage (MOTA, MOTP, IDF1, ID switches, misses, false positives).
- Result export (prediction files per sequence).

## 4) What reviewers typically check
- Reproducibility: one-click, clean run order.
- Modularity: clear separation of detector/tracker/evaluation.
- Metric integrity: same thresholds and protocol across all sequences.
- Interpretation quality: why performance is limited and what to improve.

## 5) Suggested analysis points in notebook
- Why your detector choice fits Colab runtime constraints.
- Where identity switches occur (crowds, occlusion, camera motion).
- Small-object recall issues in drone perspective.
- Threshold sensitivity (confidence and IoU).
- Trade-off between speed and accuracy.

## 6) Final pre-submission checklist
- Notebook runs top-to-bottom without manual fixes.
- Paths work with ZIP uploaded to `/content/sample_data`.
- Final metrics table is visible in output cells.
- Brief limitations + improvement plan included.
- Colab sharing permissions are open for evaluators.
- Form submission completed before the deadline.

## 7) Deadline reminder
- Mentioned deadline: **Friday, 17-04-2026, 23:59**.
- Submit earlier if possible to avoid link/permission issues.
