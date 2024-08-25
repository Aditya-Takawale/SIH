from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load YOLO with CUDA support (if available)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names file
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

caps = [cv2.VideoCapture(f'traffic_vid{i}.mp4') for i in ['', '_2', '_5', '_6']]

def gen_frames():
    start_time = time.time()
    green_signal_start_time = None
    green_signal_duration = 0

    while all(cap.isOpened() for cap in caps):
        frames = []
        traffic_counts = []

        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) < 4:
            break

        if time.time() - start_time > 10:
            break

        resized_frames = [cv2.resize(frame, (683, 384)) for frame in frames]
        output_frame = np.zeros((768, 1366, 3), dtype=np.uint8)

        output_frame[:384, :683] = resized_frames[0]
        output_frame[:384, 683:] = resized_frames[1]
        output_frame[384:, :683] = resized_frames[2]
        output_frame[384:, 683:] = resized_frames[3]

        for i, frame in enumerate(resized_frames):
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            car_count = 0
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    detection = detection.reshape(-1, 85)
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(obj[0] * width)
                            center_y = int(obj[1] * height)
                            w = int(obj[2] * width)
                            h = int(obj[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                            if classes[class_id] == "car":
                                car_count += 1

            traffic_counts.append(car_count)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

            if indexes is not None:
                for j in indexes.flatten():
                    x, y, w, h = boxes[j]
                    label = str(classes[class_ids[j]])
                    color = (0, 255, 0) if label == "car" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if i == 0:
                output_frame[:384, :683] = frame
            elif i == 1:
                output_frame[:384, 683:] = frame
            elif i == 2:
                output_frame[384:, :683] = frame
            else:
                output_frame[384:, 683:] = frame

        max_traffic = max(traffic_counts)
        max_traffic_index = traffic_counts.index(max_traffic)

        current_time = time.time()
        if green_signal_start_time is None or current_time - green_signal_start_time >= 60:
            green_signal_start_time = current_time
            green_signal_duration = 0
        else:
            green_signal_duration = current_time - green_signal_start_time

        signal_colors = [(0, 0, 255)] * 4
        signal_colors[max_traffic_index] = (0, 255, 0)

        signal_positions = [(10, 10), (1276, 10), (10, 738), (1276, 738)]
        for pos, color in zip(signal_positions, signal_colors):
            cv2.rectangle(output_frame, pos, (pos[0] + 50, pos[1] + 50), color, -1)

        cv2.putText(output_frame, f"Most Traffic: Video {max_traffic_index + 1}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    for cap in caps:
        cap.release()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if username == 'SIH' and password == 'IOIT':  # Replace with actual authentication
        return redirect(url_for('index'))
    else:
        return "Invalid credentials"

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
