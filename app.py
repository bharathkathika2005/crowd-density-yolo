import os
import cv2
import time
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Core Configurations
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure necessary directories exist based on Project Structure
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load Advanced YOLOv8x model for extreme maximum tracking
print("Loading AI Models...")
model = YOLO('yolov8x.pt')  # The largest and absolute most accurate YOLOv8 available
print("Models loaded successfully.")

video_stats = {}  # Global dictionary to track live stats for the dashboard

def process_image(filename):
    """Reads a static image, runs YOLO inference to detect crowd, and saves it."""
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error reading image {input_path}")
        return output_filename
        
    # Use ultra-high res img_sz=2560, max_det=3000, iou=0.6 and conf=0.05 to find tiny background people
    results = model(frame, classes=[0], conf=0.05, imgsz=2560, iou=0.6, max_det=3000, verbose=False)
    
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
        
    video_stats[filename] = {
        'status': 'Complete',
        'current_count': person_count,
        'max_count': person_count,
        'density': density
    }
        
    cv2.imwrite(output_path, annotated_frame)
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
    max_count = 0
    video_stats[filename] = {
        'status': 'Processing',
        'current_count': 0,
        'max_count': 0,
        'density': 'Low'
    }
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Calculate Processing Speed (FPS)
        current_time = time.time()
        fps_val = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # 2. Run Advanced YOLO inference
        # Use imgsz=1920 to scan native 1080p density.
        # conf=0.05 trusts the AI to flag anything resembling a human
        results = model(frame, classes=[0], conf=0.05, imgsz=1920, iou=0.6, max_det=3000, verbose=False)
        
        # 3. Count number of people in the frame
        person_count = 0
        if len(results) > 0:
            person_count = len(results[0].boxes)
            
        if person_count > max_count:
            max_count = person_count
            
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

        video_stats[filename]['current_count'] = person_count
        video_stats[filename]['max_count'] = max_count
        video_stats[filename]['density'] = density

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
    video_stats[filename]['status'] = 'Complete'
    print(f"Finished processing. Saved locally to {output_path}")

# ROUTES

@app.route('/api/stats/<filename>')
def get_stats(filename):
    """Returns the live counting stats of a processing video"""
    return jsonify(video_stats.get(filename, {'status': 'Unknown'}))

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

if __name__ == '__main__':
    # Running application in debug mode for development flexibility
    app.run(debug=True, port=5000)
