# Human-in-the-Loop (HITL) Object Detection with YOLOv8

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A Human-in-the-Loop (HITL) object detection pipeline built using **YOLOv8** on the **COCO128** dataset. This project introduces human oversight at key stages of the object detection workflow to improve reliability, reduce incorrect predictions, and maintain an auditable feedback process.

---

## Project Overview

Traditional object detection systems rely entirely on model predictions, which can produce incorrect or low-confidence detections. Human-in-the-Loop (HITL) systems introduce human supervision only when needed, combining the speed of AI with the reliability of human judgment.

This project demonstrates HITL at three stages:

- Pre-training label verification
- Training-time monitoring
- Post-inference prediction review

Every human action is stored in **hitl_feedback.csv**, creating a complete audit trail.

---

## Features

- Human review before model training
- Human oversight during training
- Review of low-confidence detections
- Automatic feedback logging
- Interactive widgets using ipywidgets
- Performance visualization
- YOLOv8 object detection

---

## Pipeline Architecture

```text
┌─────────────────────┐
│ Stage 1             │
│ Pre-training Review │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stage 2             │
│ Training Oversight  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stage 3             │
│ Inference Review    │
└──────────┬──────────┘
           │
           ▼
     hitl_feedback.csv
```

---

## Human Review Stages

| Stage | Trigger | Human Action |
|------|---------|--------------|
| Stage 1 | Before Training | Confirm / Flag / Skip Labels |
| Stage 2 | After Every Epoch | Continue / Stop / Reduce Learning Rate |
| Stage 3 | Low Confidence Detection | Confirm / Correct / Reject |

---

## Model Configuration

| Parameter | Value |
|------------|--------|
| Model | YOLOv8n |
| Dataset | COCO128 |
| Epochs | 3 |
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Framework | Ultralytics |
| Environment | Google Colab |

---

## Results

## Results

The YOLOv8n model was trained on the COCO128 dataset for **3 epochs** with Human-in-the-Loop supervision enabled throughout the pipeline.

### Validation Performance

| Metric | Score |
|---------|------:|
| Precision | **0.6585** |
| Recall | **0.3475** |
| mAP@50 | **0.3957** |
| mAP@50-95 | **0.2819** |

---

## Repository Structure

```text
Minor-Project/
│
├── .gitignore
├── hitl_object_detection.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/pratikagarawal/Minor-Project.git

cd Minor-Project
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook notebooks/hitl_object_detection.ipynb
```

Alternatively, open the notebook directly in **Google Colab**.

---

## How to Run

1. Open the notebook.
2. Execute the cells sequentially.
3. Complete the human review prompts.
4. Upload an image for inference.
5. Review low-confidence detections.
6. View generated graphs and feedback logs.

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- Google Colab
- OpenCV
- Matplotlib
- Pandas
- ipywidgets







