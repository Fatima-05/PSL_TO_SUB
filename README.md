# PSL Real-Time Subtitle System
## GDGOC AI/ML Bootcamp Project

### Objective
Building a real-time computer vision system that detects Pakistan Sign Language (PSL) hand gestures through a webcam and displays the corresponding Urdu alphabet as live subtitles.

### Dataset Used
Source: PSL Alphabet Dataset (SQLite database)
Features Used: 21 hand landmarks (x, y coordinates) extracted using MediaPipe, 42 input values per sample
Target Variable: Urdu alphabet letter (37 classes)

### Models Applied
MediaPipe Hands (real-time hand landmark detection)
Neural Network (Dense and BatchNorm layers for sign classification)

### Resources
- Dataset: [Pakistan Sign Language Dataset (OpenPose)](https://www.kaggle.com/datasets/saadbutt27/pakistan-sign-language-dataset-openpose) by Saad Butt
- Font: [Noto Naskh Arabic](https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic) by Google Fonts (used for Urdu subtitle rendering

## Setup

```bash
python -m venv venv
venv\Scripts\activate

pip install tensorflow-intel==2.17.0
pip install "mediapipe==0.10.14" "numpy<2.0" "opencv-python==4.8.1.78" Pillow
pip uninstall jax jaxlib -y
```

---

## Run

```bash
python app.py
```
Show your right hand to the camera. Press **Q** to quit.

The font for Urdu text downloads automatically


