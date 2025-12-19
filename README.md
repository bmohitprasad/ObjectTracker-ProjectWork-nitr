# Vehicle Tracking using YOLOv8 and Deep SORT

This project implements a real-time vehicle (car) detection, tracking, and movement analysis system using YOLOv8 for object detection and Deep SORT for multi-object tracking. The system processes a video file frame by frame, detects cars, assigns persistent identities, visualizes motion trails, computes movement metrics, calculates frames per second (FPS), and logs detailed tracking information into a CSV file for offline analysis.

The goal of this project is to demonstrate an end-to-end computer vision pipeline that combines deep learning–based object detection with appearance-based multi-object tracking for vehicle movement analysis.

------------------------------------------------------------

FEATURES

- Detects cars in video frames using YOLOv8
- Filters detections to include only the COCO car class
- Tracks multiple cars simultaneously using Deep SORT
- Assigns a unique and persistent ID to each detected car
- Maintains identity consistency across video frames
- Draws bounding boxes around tracked vehicles
- Displays the assigned car ID above each bounding box
- Computes the center point of each bounding box
- Calculates movement using Euclidean distance
- Smooths movement values using a rolling average
- Draws motion trails for visualizing vehicle paths
- Calculates real-time and averaged FPS
- Saves an annotated output video
- Logs detailed tracking data into a CSV file

------------------------------------------------------------

TECH STACK

- Python
- Ultralytics YOLOv8
- Deep SORT (deep-sort-realtime)
- OpenCV
- NumPy
- Pandas

------------------------------------------------------------

PROJECT STRUCTURE

.
├── all.mp4
├── tracked_output.mp4
├── video_tracking_output.csv
├── main.py
└── README.md

------------------------------------------------------------

INSTALLATION

Step 1: Clone the Repository

git clone https://github.com/your-username/vehicle-tracking-yolov8.git
cd vehicle-tracking-yolov8

Step 2: Install Dependencies

Ensure Python 3.8 or higher is installed.

pip install ultralytics
pip install opencv-python
pip install numpy
pip install pandas
pip install deep-sort-realtime

------------------------------------------------------------

USAGE

1. Place the input video file in the project directory.
2. Rename the video file to all.mp4
3. Run the main script.

python main.py

4. Press the q key to stop execution early if needed.

------------------------------------------------------------

OUTPUT FILES

Tracked Video Output

File name: tracked_output.mp4

The output video contains:
- Bounding boxes around detected cars
- Unique car IDs displayed above each vehicle
- Motion trails representing recent movement paths
- Same resolution and frame rate as the input video

------------------------------------------------------------

CSV Output File

File name: video_tracking_output.csv

Each row in the CSV file represents one tracked car in one video frame.

CSV Columns

frame       : Frame index of the video
timestamp   : Time when the frame was processed
car_id      : Unique ID assigned by Deep SORT
car_center  : Center of bounding box in (x, y) pixel coordinates
x1          : Left x-coordinate of bounding box
y1          : Top y-coordinate of bounding box
x2          : Right x-coordinate of bounding box
y2          : Bottom y-coordinate of bounding box
movement    : Smoothed pixel displacement between frames
fps         : Average frames per second during processing

------------------------------------------------------------

DETECTION AND TRACKING PIPELINE

1. Video frames are read sequentially using OpenCV
2. Each frame is passed to the YOLOv8 model for object detection
3. Only detections belonging to the COCO car class (class ID 2) are retained
4. Detections are converted to the format required by Deep SORT
5. Deep SORT associates detections across frames and assigns persistent IDs
6. Bounding box centers are computed for each tracked car
7. Movement is calculated using Euclidean distance
8. Movement values are smoothed using a rolling average window
9. Motion trails are drawn using historical center points
10. FPS is calculated using frame processing time
11. Annotated frames are written to the output video
12. Tracking data is logged and exported as a CSV file

------------------------------------------------------------

CUSTOMIZATION OPTIONS

- Change YOLOv8 model size for speed or accuracy
- Track different object classes by modifying the COCO class ID
- Adjust motion trail length by changing history buffer size
- Tune Deep SORT parameters such as max age and cosine distance
- Modify confidence and IoU thresholds for detection filtering

------------------------------------------------------------

USE CASES

- Traffic monitoring and analysis
- Smart city applications
- Vehicle behavior analytics
- Surveillance systems
- Computer vision research

------------------------------------------------------------

LIMITATIONS

- Movement is measured in pixel units, not real-world distance
- Performance depends on available CPU or GPU resources
- Heavy occlusion may occasionally cause ID switches
- Real-world speed estimation requires camera calibration

------------------------------------------------------------

LICENSE

This project is licensed under the MIT License.

------------------------------------------------------------

ACKNOWLEDGEMENTS

- Ultralytics YOLOv8
- Deep SORT Realtime
- OpenCV community
