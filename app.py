import os
from pathlib import Path
import cv2
import time
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
MODEL_NAME = os.environ.get('YOLO_MODEL', 'yolov8n.pt')


def resolve_model_path():
    """Return the first available local YOLO weights file, falling back to the configured model name."""
    candidates = [
        BASE_DIR / MODEL_NAME,
        PARENT_DIR / MODEL_NAME,
        BASE_DIR / 'yolov8x.pt',
        PARENT_DIR / 'yolov8x.pt',
        BASE_DIR / 'yolov8n.pt',
        PARENT_DIR / 'yolov8n.pt',
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return MODEL_NAME


model_source = resolve_model_path()

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

# Core Configurations
UPLOAD_FOLDER = str(BASE_DIR / 'uploads')
OUTPUT_FOLDER = str(BASE_DIR / 'outputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure necessary directories exist based on Project Structure
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Dictionary to hold live processing stats for the dashboard
processing_stats = {}

# Load TWO separate YOLOv8 models to perfectly balance Speed vs Accuracy
print("Loading AI Models...")
model_image = YOLO(model_source)
model_video = YOLO(model_source)
print("Models loaded successfully.")

def process_image(filename):
    """Reads a static image, runs YOLO inference to detect crowd, and saves it."""
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error reading image {input_path}")
        return output_filename
        
    if filename not in processing_stats:
        processing_stats[filename] = {"status": "Processing", "current_count": 0, "max_count": 0, "density": "Unknown"}

    # Use ultra-res img_sz=1536, low confidence (0.05), and high IOU (0.65) to allow overlapping people
    results = model_image(frame, classes=[0], conf=0.05, imgsz=1536, iou=0.65, verbose=False)
    
    person_count = 0
    if len(results) > 0:
        person_count = len(results[0].boxes)
        
    if person_count <= 10:
        density = "Low"
        color = (0, 255, 0)
    elif person_count <= 30:
        density = "Medium"
        color = (0, 255, 255)
    else:
        density = "High"
        color = (0, 0, 255)

    annotated_frame = results[0].plot()
    overlay_text = f"Count: {person_count} | Density: {density}"
    
    cv2.rectangle(annotated_frame, (10, 10), (700, 60), (0, 0, 0), -1)
    
    if person_count == 0:
        cv2.putText(annotated_frame, "No people detected in frame", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    else:
        cv2.putText(annotated_frame, overlay_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    cv2.imwrite(output_path, annotated_frame)
    
    processing_stats[filename]["current_count"] = person_count
    processing_stats[filename]["max_count"] = max(processing_stats[filename].get("max_count", 0), person_count)
    processing_stats[filename]["density"] = density
    processing_stats[filename]["status"] = "Complete"
    
    return output_filename

def process_video(filename):
    """
    Generator function that reads the video, processes each frame via YOLO,
    saves the output, and yields jpeg frames for the web stream.
    """
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    # Extract video properties to create a matching VideoWriter
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback if fps fails to read
    if fps == 0 or fps != fps: 
        fps = 30.0

    # Initialize VideoWriter to save processed video
    # 'mp4v' is a common codec for MP4 files. 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    prev_time = 0
    
    if filename not in processing_stats:
        processing_stats[filename] = {"status": "Live Processing", "current_count": 0, "max_count": 0, "density": "Unknown"}
        
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Calculate Processing Speed (FPS)
        current_time = time.time()
        fps_val = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # 2. Run YOLO inference
        # classes=[0] ensures we ONLY detect 'person' class
        # Increased imgsz to 1536, lowered conf to 0.05, and increased iou to 0.65 to allow heavily overlapping people
        results = model_video(frame, classes=[0], conf=0.05, imgsz=1536, iou=0.65, verbose=False)
        
        # 3. Count number of people in the frame
        person_count = 0
        if len(results) > 0:
            person_count = len(results[0].boxes)
            
        # 4. Classify Crowd Density
        if person_count <= 10:
            density = "Low"
            color = (0, 255, 0) # Green in BGR
        elif person_count <= 30:
            density = "Medium"
            color = (0, 255, 255) # Yellow in BGR
        else:
            density = "High"
            color = (0, 0, 255) # Red in BGR

        processing_stats[filename]["current_count"] = person_count
        processing_stats[filename]["max_count"] = max(processing_stats[filename].get("max_count", 0), person_count)
        processing_stats[filename]["density"] = density

        # 5. Draw bounding boxes around detected people
        annotated_frame = results[0].plot()
        
        # 6. Add Custom Overlays (FPS, Count, Density)
        overlay_text = f"Count: {person_count} | Density: {density} | FPS: {int(fps_val)}"
        
        # Draw a black rectangle background for the text to boost readability
        cv2.rectangle(annotated_frame, (10, 10), (700, 60), (0, 0, 0), -1)
        
        # Display Message if no people detected
        if person_count == 0:
            cv2.putText(annotated_frame, "No people detected in frame", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        else:
            cv2.putText(annotated_frame, overlay_text, (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Save the current frame to the output video sequence
        out.write(annotated_frame)
        
        # 7. Convert frame to JPEG and yield for HTTP streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Cleanup resources perfectly
    cap.release()
    out.release()
    if filename in processing_stats:
        processing_stats[filename]["status"] = "Complete"
    print(f"Finished processing. Saved locally to {output_path}")

# ROUTES

@app.route('/')
def index():
    """Renders the Home Page (Video Upload)"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the media upload via AJAX and saves it efficiently"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video or image provided'}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file requested'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        # Avoid filename collisions by adding a timestamp
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{int(time.time())}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Branch logic depending on file extension
        ext_lower = ext.lower()
        if ext_lower in ['.jpg', '.jpeg', '.png']:
            # For static images, process straight away
            process_image(unique_filename)
            redirect_url = url_for('result', filename=unique_filename, type='image')
        else:
            # For videos, redirect and load stream viewer
            redirect_url = url_for('result', filename=unique_filename, type='video')
            
        return jsonify({'redirect_url': redirect_url})

@app.route('/result/<filename>')
def result(filename):
    """Renders the Result dashboard page"""
    file_type = request.args.get('type', 'video')
    processed_filename = f"processed_{filename}"
    return render_template('result.html', filename=filename, file_type=file_type, processed_filename=processed_filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serves the saved static files (like processed images) directly from outputs/ directory"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    """Endpoint providing the multipart stream for live playback"""
    return Response(process_video(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats/<filename>')
def api_stats(filename):
    """API endpoint to provide live processing stats to the dashboard"""
    if filename in processing_stats:
        return jsonify(processing_stats[filename])
    return jsonify({"status": "Unknown", "current_count": 0, "max_count": 0, "density": "--"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=False)
