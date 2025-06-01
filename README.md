# 🔬 Surgical Tool Detection with Semi-Supervised Learning – HW1

---

## 📘 Introduction

This project focuses on the development of an object detection system tailored for the surgical environment, specifically detecting **surgical tools and hands** during leg suturing procedures.  
Given the constraints of limited labeled data, we employ **Semi-Supervised Learning (SSL)** strategies to extend our model's capability to **generalize** to **out-of-distribution (OOD)** data.

The motivation behind this project stems from recent advancements in computer vision applications within surgical settings — including robotic assistance, surgical phase recognition, and surgeon training systems.

---

## 🎯 Objective

Develop an object detection system that:
1. **Input**: Static image or video  
2. **Output**: Bounding boxes around hands and tools  
   - Tools are categorized as: **Tweezers**, **Needle Driver**, or **Empty**

---

## 🚀 Methodology

### 🔹 Model

We utilize **YOLOv8n** due to its optimal tradeoff between speed and accuracy.

---

### 🔹 Experimental Pipeline

#### **Stage 1: Baseline**
- Trained YOLOv8n on the **original 61 labeled images** (unaltered).

#### **Stage 2: Data Augmentation**
- Applied the following augmentations to both training and validation sets:
  - Horizontal Flip
  - Rotation
  - Brightness Adjustment
  - Scaling
  - Blurring
  - Gaussian Noise
- Resulted in a significantly larger training dataset.
- Trained a new model on this **augmented + original** data.

#### **Stage 3: Semi-Supervised Learning (In-Distribution)**
- Extracted frames from **in-distribution (ID)** surgery videos.
- Used the Stage 2 model to generate predictions.
- Selected high-confidence predictions (**confidence > 0.85**) as **pseudo-labels**.
- Added these pseudo-labeled frames to the training set.

#### **Stage 4: Augmenting Pseudo-Labeled Data**
- Applied the same augmentations from Stage 2 to the pseudo-labeled images from Stage 3.
- Combined with previous datasets and trained a stronger model.

#### **Stage 5: Incorporating OOD Data**
- Extracted frames from the **out-of-distribution (OOD)** surgery video.
- Used the Stage 4 model to predict bounding boxes.
- Retained predictions with **confidence > 0.85** as additional pseudo-labeled images.
- Trained a **final model** on this enriched dataset.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/surgical-tool-detection.git
cd surgical-tool-detection
```

### 2. Create a Python Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> **Note:** Requires Python 3.8 or later.  
> YOLOv8 requires `ultralytics`, which is included in `requirements.txt`.

---

## 📸 Usage

This project can be used for surgical tools detection on **images** and **videos**.

---

### 🖼️ Predicting on an Image

The `predict.py` script is used to detect surgical tools and hands in a single image.

#### Instructions:

1. Download `weights.pt` from this repository.
2. Open the `predict.py` file.
3. Modify the following lines in the `main()` function:

```python
weights_path = "path/to/your/weights.pt"
image_path = "path/to/your/image.jpg"
output_path = "output.jpg"
```

4. To **create an output image**, set `create_output_image=True`:
```python
predictions = predict_on_image(model, image_path, output_path, create_output_image=True)
```

5. To **generate a prediction file only** in `(x_center, y_center, width, height, confidence, class)` format, set:
```python
predictions = predict_on_image(model, image_path, output_path, create_output_image=False)
```

> Output files will be saved in the project directory.

---

### 🎥 Predicting on a Video

The `video.py` script is used for detecting tools and hands in surgical videos.

#### Instructions:

1. Download `weights.pt` from this repository.
2. Open the `video.py` file.
3. Modify the following in the `main()` function:

```python
weights_path = "path/to/your/weights.pt"
video_path = "path/to/your/video.mp4"
output_path = "path/to/save/prediction/files"
```

4. To **generate an output video**, set `generate_output_video=True`:
```python
predict_on_video(model, video_path, output_path, generate_output_video=True)
```

5. To **generate prediction files only**, set:
```python
predict_on_video(model, video_path, output_path, generate_output_video=False)
```

> - The output video will be saved in the project folder.  
> - Prediction files (in YOLO format) will be saved under a folder called `frame_predictions` at the specified `output_path`.

