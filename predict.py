# predict_yolo_image.py

import cv2
from ultralytics import YOLO

def predict_on_image(
    model,
    image_path,
    output_path,
    create_output_image=False,
    output_image_path="image_with_boxes.jpg",
    conf_threshold=0.5
):
    """
    Runs YOLO model on a single image, saves YOLO-format labels to .txt,
    and optionally saves an image with bounding boxes.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    # Run prediction
    results = model(image)[0]

    # Filter predictions by confidence threshold
    predictions = []
    for box in results.boxes:
        conf = float(box.conf)
        if conf >= conf_threshold:
            class_id = int(box.cls)
            x_center, y_center, w, h = box.xywhn[0].tolist()  # Normalized values
            predictions.append((class_id, conf, x_center, y_center, w, h))

    # Save predictions to YOLO-format .txt file (without confidence)
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]} {pred[2]} {pred[3]} {pred[4]} {pred[5]}\n")

    # Optionally save image with bounding boxes
    if create_output_image:
        annotated = results.plot()
        cv2.imwrite(output_image_path, annotated)

    print(f"✅ Saved {len(predictions)} labels to {output_path}")
    if create_output_image:
        print(f"🖼️ Saved annotated image to {output_image_path}")

    return predictions

if __name__ == "__main__":
    # Define paths
    weights_path = "weights.pt"  # path to trained YOLO model - do not change
    image_path = "test.jpg" #change to your path
    output_path = "label_output.txt"
    output_image_path = "image_output.jpg"

    # Load model
    model = YOLO(weights_path)

    # Run prediction
    predictions = predict_on_image(
        model=model,
        image_path=image_path,
        output_path=output_path,
        create_output_image=True,
        output_image_path=output_image_path
    )