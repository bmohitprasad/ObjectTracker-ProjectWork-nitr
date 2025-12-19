import cv2
import time
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Input / Output Paths ===
VIDEO_PATH = "all.mp4"              # <-- put your video file here
OUTPUT_VIDEO_PATH = "tracked_output.mp4"
CSV_OUTPUT_PATH = "video_tracking_output.csv"

# === Load YOLOv8 Model ===
model = YOLO('yolov8l.pt')   # or 'yolov8s.pt' if you want faster inference

# === Initialize Deep SORT Tracker ===
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4
)

# === Open Video File ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Failed to open video file.")
    raise SystemExit

# === Get Video Properties ===
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input    = cap.get(cv2.CAP_PROP_FPS) or 20.0

# === Video Writer for Output ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_input, (frame_width, frame_height))

# === Tracking & Logging State ===
frame_data = []
fps_list = deque(maxlen=10)
prev_centers = {}      # track_id -> last center
movement_history = {}  # track_id -> deque of last N movements
trace_history = {}     # track_id -> deque of last N centers (for trails)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # finished video

    start_time = time.time()

    # === YOLOv8 Detection ===
    results = model(frame, conf=0.25, iou=0.45)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # Only track cars (COCO class id 2)
        if cls_id == 2:
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'car'))

    # === Deep SORT Tracking ===
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        center = ((l + r) // 2, (t + b) // 2)

        # --- Movement computation ---
        movement = 0.0
        if track_id in prev_centers:
            dx = center[0] - prev_centers[track_id][0]
            dy = center[1] - prev_centers[track_id][1]
            movement = (dx**2 + dy**2) ** 0.5

        prev_centers[track_id] = center

        # Smooth movement per ID
        if track_id not in movement_history:
            movement_history[track_id] = deque(maxlen=5)
        movement_history[track_id].append(movement)
        smoothed_movement = float(np.mean(movement_history[track_id]))

        # Store trace for trails
        if track_id not in trace_history:
            trace_history[track_id] = deque(maxlen=30)
        trace_history[track_id].append(center)

        # FPS measurement
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        fps_list.append(fps)
        avg_fps = sum(fps_list) / len(fps_list)

        # --- Draw bounding box and ID ---
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'Car ID: {track_id}', (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Draw motion trail ---
        trail = trace_history[track_id]
        for i in range(1, len(trail)):
            pt1 = trail[i - 1]
            pt2 = trail[i]
            if pt1 and pt2:
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # --- Log frame data ---
        frame_data.append({
            "frame": frame_id,
            "timestamp": round(time.time(), 2),
            "car_id": track_id,
            "car_center": str(center),
            "x1": l,
            "y1": t,
            "x2": r,
            "y2": b,
            "movement": round(smoothed_movement, 2),
            "fps": round(avg_fps, 2)
        })

    # === Show and Save Frame ===
    cv2.imshow("Recorded Video - YOLOv8 + Deep SORT + Trails", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === Save CSV ===
df = pd.DataFrame(frame_data)
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Done.\nVideo saved as: {OUTPUT_VIDEO_PATH}\nCSV saved as: {CSV_OUTPUT_PATH}")
