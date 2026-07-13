# Human-in-the-Loop (HITL) Object Detection with YOLOv8

A minor project exploring **Human-in-the-Loop machine learning** — where a human reviewer
is inserted at key decision points of an object detection pipeline instead of letting the
model run fully automatically. Built with **YOLOv8 (Ultralytics)** on the **COCO128**
dataset, in Google Colab.

The core idea: full automation is risky when the model is uncertain or wrong, but full
manual review is too slow. This project puts a human in the loop *only* at the points
where it matters most — flagging bad labels before training, watching for performance
drops during training, and reviewing low-confidence predictions after inference — and
logs every human decision for auditability.

## Why Human-in-the-Loop?

Fully automated pipelines silently accept whatever the model predicts, including
confidently-wrong or low-confidence detections. HITL systems instead route uncertain or
risky cases to a human, and use that feedback to improve the pipeline. This project
implements three separate HITL checkpoints across the model lifecycle:

| Stage | When | What triggers human review | Human's decision |
|---|---|---|---|
| **1 — Pre-training label review** | Before training starts | Every sampled image is shown with its ground-truth boxes | Confirm label / flag as wrong / skip |
| **2 — During-training oversight** | After every training epoch | mAP drop greater than a threshold, or simply every epoch | Continue / stop / halve learning rate |
| **3 — Post-inference review** | After running inference | Any detection with confidence below a threshold | Confirm label / correct label / reject detection |

Every human decision is logged to `hitl_feedback.csv` (stage, image, predicted label,
confidence, human decision, action taken, updated confidence), so the human oversight
itself is fully auditable.

## Pipeline architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│   STAGE 1            │     │   STAGE 2              │     │   STAGE 3               │
│   Pre-training        │ →   │   During-training       │ →   │   Post-inference         │
│   label review         │     │   oversight              │     │   confidence review       │
│                        │     │                          │     │                          │
│ Human confirms/flags   │     │ Custom Ultralytics       │     │ Detections below         │
│ ground-truth labels    │     │ callback checks mAP      │     │ CONFIDENCE_THRESH are     │
│ before training runs   │     │ after every epoch;       │     │ shown to a human for      │
│                        │     │ human can continue/       │     │ confirm / correct /       │
│                        │     │ stop/halve LR             │     │ reject                    │
└─────────────────────┘     └──────────────────────┘     └───────────────────────┘
                    all decisions logged → hitl_feedback.csv
```

## Results

Trained YOLOv8n on COCO128 for 3 epochs (image size 224, batch 32) with human oversight
active at every epoch:

| Metric | Value |
|---|---|
| Precision | 0.659 |
| Recall | 0.348 |
| mAP@50 | 0.396 |
| mAP@50-95 | 0.282 |

11 human interventions were logged across all three stages during this run (label
confirmations/flags in Stage 1, epoch continue/stop decisions in Stage 2, and detection
confirm/correct/reject decisions in Stage 3).



1. **Action distribution** — breakdown of human decisions (confirm / correct / reject / etc.)
2. **Confidence before vs. after human review** — shows the effect of human correction on low-confidence detections
3. **Interventions per stage** — how many human decisions happened at each of the 3 stages
4. **Confidence improvement delta** — per-detection change in confidence after human review
5. **mAP per epoch** — training curve with the HITL drop-alert threshold marked
6. **Post-inference confidence distribution** — histogram of detection confidence, before and after human correction

## Repository structure

```
.
├── notebooks/
│   └── hitl_object_detection.ipynb   # Main notebook (run outputs included)
├── assets/                           # Saved output graphs, used in this README
├── requirements.txt
└── README.md
```

## Getting started

This project was built and run in **Google Colab**. To reproduce it:

1. Open `notebooks/hitl_object_detection.ipynb` in Google Colab, **or** run it locally
   in Jupyter with:
   ```bash
   pip install -r requirements.txt
   jupyter notebook notebooks/hitl_object_detection.ipynb
   ```
2. Run the cells in order, top to bottom. Several cells are designed to be **re-run
   manually** (e.g., once per image in Stage 1) — this is intentional; the notebook
   prints how many times to re-run each cell.
3. Stage 3 expects you to upload an image for inference when prompted.

> **Note on interactivity:** the human review steps use `ipywidgets` (dropdowns and
> confirm buttons), which render as live, clickable widgets in Colab/Jupyter but only
> as a static placeholder in GitHub's notebook preview. The saved outputs (banners,
> images, graphs) are still visible on GitHub either way — to interact with the review
> widgets yourself, open the notebook in Colab.

## Tech stack

- **YOLOv8n** (Ultralytics) — object detection model
- **COCO128** — small subset of COCO used for fast iteration
- **ipywidgets** — interactive human review widgets
- **OpenCV / Matplotlib** — image handling and visualization
- **pandas** — feedback logging and analysis

## Possible extensions

- Persist human corrections back into the training set for a true active-learning loop
- Replace the fixed confidence threshold with a learned/adaptive threshold
- Add inter-annotator agreement tracking if multiple humans review the same detections
- Package the non-widget logic into a standalone Python module for use outside notebooks

