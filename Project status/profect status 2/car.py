from flask import Flask, Response
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np
import multiprocessing
import time

app = Flask(__name__)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (320, 320)}))
picam2.start()

frame_queue = multiprocessing.Queue(maxsize=1)
result_queue = multiprocessing.Queue(maxsize=1)

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output


def yolo_detection_process(frame_queue, result_queue, pid_controller):
    model = YOLO("BFMCv11_ncnn_model", task='segment')
    target_cX = 160  
    prev_time = time.time()

    while True:
        try:
            frame = frame_queue.get(timeout=1)  
        except:
            continue  

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(source=frame, show=False, save=False, conf=0.6, imgsz=320,verbose=False)

        annotated_frame = results[0].plot()

        cX = 160 
        for result in results:
            for c in result:
                label_index = c.boxes.cls[0]
                if label_index == 9:  
                    b_mask = np.zeros(annotated_frame.shape[:2], np.uint8)
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                    M = cv2.moments(b_mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(annotated_frame, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, "Center", (cX - 10, cY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        error = target_cX - cX 
        pid_output = int(pid_controller.update(error, dt))

        # servo + pid_output

        pid_text = f"PID Output: {pid_output}"
        cv2.putText(annotated_frame, pid_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 30), 2)

        result_queue.put((annotated_frame, cX))


def generate_frames():
    """Generate frames for the video stream."""
    while True:
        frame = picam2.capture_array()

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        try:
            if frame_queue.full():
                frame_queue.get_nowait()  
            frame_queue.put_nowait(frame)
        except:
            continue  

        try:
            annotated_frame, _ = result_queue.get(timeout=1)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except:
            continue  


@app.route('/video_feed')
def video_feed():
    """Endpoint for video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Main page."""
    return "<h1>RPi Camera Stream</h1><img src='/video_feed'>" 


if __name__ == "__main__":
    pid_controller = PIDController(Kp=1, Ki=0.00, Kd=0.00)

    yolo_process = multiprocessing.Process(target=yolo_detection_process, args=(frame_queue, result_queue, pid_controller))
    yolo_process.start()

    try:
        app.run(host='0.0.0.0', port=5050)
    except KeyboardInterrupt:
        print("Shutting down...")

    yolo_process.terminate()
    yolo_process.join()
