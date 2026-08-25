# Crowd Density Estimation using YOLO and OpenCV

This is a complete web-based system that detects people in a video, counts them in real-time, and classifies the crowd density.

## Technologies Used
- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Frontend:** HTML, CSS (Vanilla Custom CSS), JavaScript

## Setup Instructions

1. **Prerequisites:**
   Ensure you have Python 3.8+ installed on your system.

2. **Install Dependencies:**
   Open a terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```
   > Note: `ultralytics` will automatically download the default `yolov8n.pt` weights file the first time you run the application. To use another YOLO model, set the `YOLO_MODEL` environment variable, for example `YOLO_MODEL=yolov8x.pt`.

3. **Run the Application:**
   Execute the `app.py` script to start the local server:
   ```bash
   python app.py
   ```

## Deploying

The included `Procfile` is ready for platforms such as Render, Railway, and Heroku. Use the following build and start commands when the platform asks for them:

```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 app:app
```

The server listens on the platform-provided `PORT` and binds to all interfaces. Keep workers at `1` because each worker loads the YOLO model into memory. The `uploads/` and `outputs/` folders are temporary on most hosting platforms, so use persistent storage or object storage if processed files must survive a restart.

4. **Access the Web App:**
   Open your web browser and go to: `http://127.0.0.1:5000`

## Workflow Diagram

```text
[ User ] --(Uploads Video)--> [ Web Server / Flask (`/`) ]
                                      |
                                      v
                             [ `uploads/` Folder ]
                                      |
                                      v
[ Web Browser (`/result`) ] <-----(Redirect)
        |
        +--(Requests stream)--> [ Flask (`/video_feed`) ]
                                      |
                                      v
                             [ OpenCV extracts Frame ]
                                      |
                                      v
[ OpenCV saves to `outputs/` ] <-- [ YOLOv8 detects 'person' ]
                                      |
                                      v
                             [ Draw Bounding Boxes + Density ]
                                      |
                                      v
[ Web Browser Stream ] <----(Yields JPEG Frame)----
```

## Step-by-step Explanation of the Logic
1. **Frontend Upload:** The user drags and drops a video on `index.html`. Using JavaScript (AJAX), the video is sent to the `/upload` backend route with a continuous progress bar tracking the upload.
2. **File Saving:** Flask temporarily saves the source video into `uploads/`. A unique timestamp is added to prevent overwrites.
3. **Stream Initiation:** The user is redirected to `result.html`, which contains an `<img>` tag pointing to the `/video_feed/<filename>` route.
4. **Frame Processing:** The `/video_feed` is a generator. It opens the saved video via `cv2.VideoCapture`. 
   For every frame:
   - We calculate frames-per-second (FPS) processing speed.
   - We run YOLOv8 tracking strictly class id `0` (person).
   - We check the count of detected bounding boxes.
   - We classify density (0-10: Low, 11-30: Medium, 31+: High).
   - We render custom rectangles and text on the frame.
5. **Simultaneous File Saving:** Every annotated frame is written to an `.mp4` file in `outputs/` via `cv2.VideoWriter`.
6. **Live Viewing:** Simultaneously, the exact frame is encoded to JPEG format and yielded back to the browser via `multipart/x-mixed-replace` MIME type.
7. **End of Video:** Output streams are closed, memory released.
