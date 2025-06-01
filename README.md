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

Throughout the project, we employed the **YOLOv8n** model as the primary object detection architecture. The experimental procedure was structured into **five progressive stages**, each designed to systematically improve model performance by enriching the training data.

---

### 🔹 Stage 1: Baseline Model
We began by training a baseline model on the original dataset, which consisted of **61 labeled training images**. This model served as a reference point to evaluate the impact of subsequent dataset enhancements.

---

### 🔹 Stage 2: Data Augmentation
To improve generalization and robustness, we expanded the dataset by generating **six synthetic variants per image** for the 61 training images and 10 validation images.  
Augmentation techniques included:
- Horizontal flipping  
- Rotation  
- Brightness adjustment  
- Scaling  
- Blurring  
- Gaussian noise  

The model was retrained on the combined set of original and augmented images.

---

### 🔹 Stage 3: Semi-Supervised Learning on In-Distribution Videos
In this stage, we leveraged **semi-supervised learning (SSL)** to incorporate unlabeled frames from **in-distribution (ID) videos**. The model from Stage 2 generated predictions on extracted video frames.  
Only frames with **confidence scores > 0.7** were selected and added as **pseudo-labeled data**.

- The training set now included:
  - Original and augmented images from Stage 2 (`X` images)
  - High-confidence pseudo-labeled frames from ID videos (`Y` images)
  - **Total: Z images**

This enriched dataset was split into new training and validation sets, and a model was retrained.

---

### 🔹 Stage 4: Incorporation of Out-of-Distribution Data
We further enhanced the dataset by including frames from an **out-of-distribution (OOD) video** (recorded on a different day with a different camera setup).  
The model from Stage 3 predicted on these frames, and we again selected **only high-confidence (>0.7) predictions**. These were added to the training set.

A final model was trained on this comprehensive and diverse dataset, better equipping it for real-world generalization.

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

