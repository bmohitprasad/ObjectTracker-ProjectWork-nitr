# Vehicle Tracking using YOLOv8 and Deep SORT

This project implements a real-time vehicle (car) detection, tracking, and movement analysis system using YOLOv8 for object detection and Deep SORT for multi-object tracking. The system processes a video file, tracks cars frame-by-frame, visualizes motion trails, calculates movement and FPS, and logs tracking data into a CSV file.

---

## Features

- Car detection using YOLOv8 (COCO class: car)
- Multi-object tracking using Deep SORT
- Persistent unique ID assignment per vehicle
- Motion trail visualization for tracked cars
- Smoothed movement estimation per vehicle
- Real-time FPS calculation
- Annotated output video generation
- Frame-wise CSV logging for analysis

---

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- Deep SORT (deep-sort-realtime)
- OpenCV
- NumPy
- Pandas

---

## Project Structure

├── all.mp4
├── tracked_output.mp4
├── video_tracking_output.csv
├── main.py
└── README.md


---

## Installation

### Clone the Repository
git clone https://github.com/your-username/vehicle-tracking-yolov8.git
cd vehicle-tracking-yolov8

Install Dependencies

Ensure Python 3.8 or later is installed.

pip install ultralytics opencv-python numpy pandas deep-sort-realtime

Usage

Place the input video file in the project directory and name it all.mp4.

Run the tracking script:

python main.py


Press q to stop execution at any time.

Output
Tracked Video

File: tracked_output.mp4

Contains bounding boxes, unique car IDs, motion trails, and FPS overlay.

CSV Output

File: video_tracking_output.csv

Columns:

frame

timestamp

car_id

car_center

x1, y1, x2, y2

movement

fps

The CSV file can be used for traffic analysis, vehicle movement studies, or visualization pipelines.

Detection and Tracking Pipeline

YOLOv8 performs object detection on each frame.

Only cars (COCO class ID = 2) are selected.

Deep SORT assigns persistent IDs and maintains identity across frames.

Vehicle movement is calculated using Euclidean distance between consecutive bounding box centers.

Movement values are smoothed using a rolling average to reduce noise.

Customization

Change YOLO model size:

model = YOLO('yolov8s.pt')  # Faster inference
model = YOLO('yolov8l.pt')  # Higher accuracy


Track other object classes by modifying the COCO class ID filter.

Adjust motion trail length by changing the deque size.

Use Cases

Traffic monitoring and analysis

Smart city applications

Vehicle behavior analytics

Surveillance systems

Computer vision research projects

Limitations

Movement is measured in pixel units, not real-world distance.

Tracking accuracy may degrade under heavy occlusion.

Performance depends on available hardware resources.

License

This project is licensed under the MIT License.

Acknowledgements

Ultralytics YOLOv8

Deep SORT Realtime

OpenCV community
