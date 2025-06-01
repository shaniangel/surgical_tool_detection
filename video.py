import cv2
import os
from ultralytics import YOLO

def predict_on_video(
    model,
    video_path,
    output_path,
    conf_threshold=0.5,
    imgsz=640,
    class_names=None,
    colors=None
):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Cannot open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Found video with {total_frames} frames at {fps} FPS.")

    # Output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run prediction
        results = model.predict(frame, imgsz=imgsz, conf=conf_threshold)[0]

        # Annotate each box
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xc, yc, bw, bh = box.xywh[0]
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)

            color = (0, 255, 0) if colors is None else colors[cls % len(colors)]
            label = f"{class_names[cls]} ({conf:.2f})" if class_names else f"Class {cls} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Saved annotated video to: {output_path}")

if __name__ == "__main__":
    weights_path = "weights.pt"  # path to trained YOLO model - do not change
    model = YOLO(weights_path)

    predict_on_video(
        model=model,
        video_path="surg_1.mp4",  # change to your video path
        output_path="./surg_1_annotated.mp4",        # change to desired output path
        conf_threshold=0.4,
        class_names=["Empty", "Tweezers", "Needle Driver"],
        colors=[(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    )