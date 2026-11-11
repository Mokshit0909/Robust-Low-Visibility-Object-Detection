# Robust Object Detection in Low-Visibility Environments

An experimental computer vision project focused on object detection in foggy and blurred environments using GroundingDINO, synthetic fog augmentation, and video-based inference.

## Project Overview

Object detection becomes challenging in adverse weather conditions such as fog, where reduced visibility and image quality can affect the detection of road users.

This project explores a low-visibility detection workflow using synthetic fog and blur augmentation, text-prompt-based object detection, confidence filtering, and temporal detection stability.

## Key Components

### 1. Synthetic Fog and Blur Augmentation

Implemented using OpenCV and NumPy.

- Simulates fog using a white overlay.
- Applies Gaussian blur to reduce image clarity.
- Processes Cityscapes validation images.
- Saves the generated images for experimentation.

Main scripts:
- `demo/augment_fog.py`
- `demo/fog_augmentation.py`

### 2. Text-Prompt-Based Object Detection

Uses GroundingDINO for open-vocabulary object detection.

Example detection prompts include:
- Person
- Car
- Bus
- Scooter
- Truck
- Rickshaw

The detection configuration uses confidence thresholds to filter predictions.

### 3. Video-Based Object Detection

The video inference implementation includes:

- CPU-based model inference.
- Frame skipping.
- Confidence-based detection filtering.
- Temporal persistence of previous detections for a limited number of missing frames.
- Bounding-box annotation.
- FPS overlay and video output.

Main script: `demo/inference_on_video.py`

### 4. Interactive Image Detection

A Gradio-based interface is included for image detection using text prompts and GroundingDINO.

Main script: `demo/gradio_app.py`

## Technology Stack

- Python
- PyTorch
- GroundingDINO
- OpenCV
- NumPy
- Gradio
- Cityscapes dataset

## Repository Structure

```text
Robust-Low-Visibility-Object-Detection/
│
├── .asset/
├── demo/
│   ├── augment_fog.py
│   ├── fog_augmentation.py
│   ├── inference_on_video.py
│   ├── gradio_app.py
│   └── video_to_frames.py
│
├── groundingdino/
├── GROUNDINGDINO_README.md
├── requirements.txt
├── environment.yaml
├── setup.py
├── Dockerfile
├── LICENSE
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mokshit0909/Robust-Low-Visibility-Object-Detection.git
cd Robust-Low-Visibility-Object-Detection
```

### 2. Install dependencies

Create and activate a Python environment compatible with the project's dependencies.

```bash
pip install -r requirements.txt
```

Install GroundingDINO using the repository setup instructions:

```bash
pip install -e .
```

### 3. Download model weights

The pretrained GroundingDINO checkpoint is not included in this repository.

Download the appropriate checkpoint and place it in a `weights/` directory. Ensure that the checkpoint filename and model configuration match the paths used in the inference script.

### 4. Prepare the dataset and input video

The Cityscapes dataset and sample videos are not included in this repository.

Place the dataset and input video in the expected locations before running the corresponding scripts.

## Running the Project

Run the video inference script from the `demo` directory after preparing the model weights and input video:

```bash
cd demo
python inference_on_video.py
```

Run fog augmentation from the `demo` directory after preparing the Cityscapes validation dataset:

```bash
python fog_augmentation.py
```

For the interactive Gradio application, check the configuration and checkpoint paths in `demo/gradio_app.py` before launching it.

## Notes and Limitations

- The project uses synthetic fog and Gaussian blur for augmentation.
- Detection performance has not been quantified in this repository documentation.
- Pretrained model weights and datasets must be obtained separately.
- CPU inference may be slower than GPU inference.
- Input paths and model configuration paths may need adjustment for your local environment.

## Acknowledgements

This project uses GroundingDINO, developed by IDEA Research.

Original repository:
https://github.com/IDEA-Research/GroundingDINO

Research paper:
https://arxiv.org/abs/2303.05499

Please refer to the original GroundingDINO repository and its license for model implementation, licensing, and attribution details.

---

Developed as an academic computer vision project.
