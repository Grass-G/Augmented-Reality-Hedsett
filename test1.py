from ultralytics import YOLO
import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template_string
import threading
import time
import ollama
from threading import Lock

app = Flask(__name__)

LLM_MODEL = "deepseek-r1:7b"
llm_lock = Lock()
current_description = "No object detected"

CAMERA_URL = "http://192.168.1.8:4747/video"
MODEL_PATH = "yolo-Weights/yolov8n.pt"
IMG_SIZE = 640
CONF_THRESH = 0.5
HALF_PRECISION = True


latest_frame = None
latest_frame_time = None
frame_lock = threading.Lock()
fps_lock = threading.Lock()
model = None
fps = 0
frame_count = 0
start_time = time.time()

@app.route('/get_desc')  # Fixed endpoint name
def get_desc():
    return Response(current_description, mimetype='text/plain')

def get_object_description(object_name):
    """Get concise description using Ollama client"""
    global current_description
    
    prompt = (
        f"Describe '{object_name}' in 25 words. Be technical and concise. "
        "Format: 'OBJECT: Description...'"
    )
    
    try:
        with llm_lock:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={'temperature': 0.1}
            )
            desc = response['message']['content'].strip()
            current_description = desc.split("\n")[0]  # Take first line only
            print(desc)
            
    except Exception as e:
        current_description = f"{object_name}: Description unavailable"
        print(f"LLM Error: {str(e)}")

def process_detection(object_name):
    # Run in background thread to avoid blocking video pipeline
    import threading
    threading.Thread(target=get_object_description, args=(object_name,)).start()


HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>RTX Object Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <style>
        body { 
            margin: 0; 
            background: #111;
            height: 100vh;
            padding: 2.5%;  /* Reduced outer padding */
            box-sizing: border-box;
        }
        .container {
            display: flex;
            height: 100%;
            gap: 5%;  /* Reduced gap */
            justify-content: space-between;
        }
        .video-container {
            flex: 0 0 45%;  /* Fixed width with flex-basis */
            position: relative;
            overflow: hidden;
            background: black;
            border-radius: 8px;
            aspect-ratio: 16/9;  /* Maintain aspect ratio */
        }
        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .crosshair {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 30px;
            height: 30px;
            pointer-events: none;
        }
        .crosshair::before,
        .crosshair::after {
            content: '';
            position: absolute;
            background: #00ffff;
        }
        .crosshair::before {
            width: 100%;
            height: 2px;
            top: 50%;
            transform: translateY(-1px);
        }
        .crosshair::after {
            height: 100%;
            width: 2px;
            left: 50%;
            transform: translateX(-1px);
        }
        .hud {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #00ffff;
            font-family: monospace;
            text-shadow: 1px 1px 2px #000;
            z-index: 2;
            background: rgba(0,0,0,0.5);
            padding: 5px;
            border-radius: 3px;
        }
        .ai-desc {
            color: #00ff00;
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
            max-width: 300px;
            font-family: monospace;
        }

        @media (orientation: portrait) {
            body {
                padding: 2.5%;
            }
            .container {
                flex-direction: column;
                gap: 5%;
            }
            .video-container {
                flex-basis: 45vh;  /* Fixed height in portrait */
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-container">
            <div class="hud">
                <div>FPS: <span id="fps">-</span></div>
                <div>TIME: <span id="time">-</span></div>
                <div>LATENCY: <span id="latency">-</span>ms</div>
            </div>
            <div class="crosshair"></div>
            <img class="video-feed" src="/video_feed">
        </div>
        <div class="video-container">
            <img class="video-feed" src="/video_feed">
        </div>
    </div>
    <div class="ai-desc" id="ai-desc"></div>
    <script>
        // Remove any external script references
        async function updateHUD() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                // Update first feed
                document.getElementById('fps').textContent = data.fps;
                document.getElementById('time').textContent = data.time;
                document.getElementById('latency').textContent = data.latency;
                // Update second feed
                document.getElementById('fps2').textContent = data.fps;
                document.getElementById('time2').textContent = data.time;
                document.getElementById('latency2').textContent = data.latency;
            } catch (error) {
                console.log('HUD update error:', error);
            }
        }
        setInterval(updateHUD, 1000);
function updateDesc() {
        fetch('/get_desc')  // Matching endpoint
            .then(r => r.text())
            .then(t => document.getElementById('ai-desc').textContent = t);
    }
    setInterval(updateDesc, 1000);
    </script>
</body>
</html>
'''
def camera_thread():
    global model, latest_frame, latest_frame_time, fps, frame_count, start_time
    
    try:
        model = YOLO(MODEL_PATH).to('cuda')
        model.fuse()
        if HALF_PRECISION:
            model.half()
        

        _ = model.predict(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to('cuda').half(), verbose=False)
        
        cap = cv2.VideoCapture(CAMERA_URL)
        if not cap.isOpened():
            raise RuntimeError(f"Camera connection failed: {CAMERA_URL}")
        
        print("Camera initialized successfully!")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error, retrying...")
                time.sleep(1)
                continue


            capture_time = time.time()
            img_tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) \
                .to('cuda', non_blocking=True) \
                .half() if HALF_PRECISION else float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                results = model.predict(img_tensor, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]


            for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                    results.boxes.cls.cpu().numpy().astype(int),
                                    results.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                label = f"{results.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            with frame_lock:
                latest_frame = frame.copy()
                latest_frame_time = capture_time

            with fps_lock:
                frame_count += 1
                if (time.time() - start_time) >= 1:
                    fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()

    except Exception as e:
        print(f"Camera thread crashed: {str(e)}")

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_data = buffer.tobytes() if ret else b''
            else:

                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with fps_lock:
        fps_value = round(fps, 1) if fps > 0 else 0.0
    
    with frame_lock:
        latency_value = round((time.time() - latest_frame_time)*1000, 1) if latest_frame_time else 0.0
    
    return {
        'fps': fps_value,
        'time': time.strftime("%H:%M:%S"),
        'latency': latency_value
    }

if __name__ == '__main__':
    print("Starting server...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    threading.Thread(target=camera_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)

