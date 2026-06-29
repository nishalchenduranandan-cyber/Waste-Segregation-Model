from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "my_trained_yolov8_model2.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


def run_inference(image_bytes: bytes, conf_threshold: float = 0.30):
    """Run YOLOv8 inference on an image uploaded via Streamlit.

    Returns the original image, annotated image, and a list of detection strings.
    """
    if not image_bytes:
        raise ValueError("No image data provided.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    original_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError("Unable to decode uploaded image.")

    results = model(original_image, conf=conf_threshold, imgsz=640)
    annotated_image = results[0].plot()

    detections = []
    detection_counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        conf = float(box.conf.item())
        detections.append(f"{label}: {conf:.2f}")
        detection_counts[label] = detection_counts.get(label, 0) + 1

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"inference_{uuid4().hex}.jpg"
    cv2.imwrite(str(output_path), annotated_image)

    success, encoded_image = cv2.imencode('.jpg', annotated_image)
    if not success:
        raise RuntimeError('Failed to encode annotated image for download.')

    annotated_bytes = encoded_image.tobytes()
    return original_image, annotated_image, annotated_bytes, detections, detection_counts
