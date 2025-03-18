import serial
import time
import multiprocessing
from flask import Flask, Response
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np

app = Flask(__name__)

# Define Serial Port Parameters Globally
serial_port = '/dev/ttyACM0'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (320, 320)}))
picam2.start()

frame_queue = multiprocessing.Queue(maxsize=1)
result_queue = multiprocessing.Queue(maxsize=1)

# PID Controller Class
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


def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have 
        identified edges in the frame
    """
    # create an array of the same size as of the input image 
    mask = np.zeros_like(image)   
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
          # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.1, rows * 0]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.9, rows * 0]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
 
def hough_transform(image):
    if image is None or np.count_nonzero(image) == 0:
        return None  # No lines to detect if the image is empty or doesn't have enough edges
    rho = 1             
    theta = np.pi / 180   
    threshold = 20      
    minLineLength = 20  
    maxLineGap = 500    
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
     
def average_slope_intercept(lines):
    if lines is None:
        return None, None  # Return None for both lanes if no lines are detected
    left_lines = [] #(slope, intercept)
    left_weights = [] #(length,)
    right_lines = [] #(slope, intercept)
    right_weights = [] #(length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
   
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    try: x1 = int((y1 - intercept)/slope)
    except: x1 = 0
    try: x2 = int((y2 - intercept)/slope)
    except: x2 = 0
    try: y1 = int(y1)
    except: y1 = 0
    try: y2 = int(y2)
    except: y2 = 0
    return ((x1, y1), (x2, y2))
   
def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    if left_lane is None or right_lane is None:
        return None, None  # If no valid lines, return None
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
 
     
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
 
def frame_processor(image):
    """
    Process the input frame to detect lane lines and return intermediate frames for debugging.
    Parameters:
        image: image of a road where one wants to detect lane lines
        (we will be passing frames of video to this function)
    """
    # Convert the RGB image to Grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to remove noise
    kernel_size = 7
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    
    # First threshold for Canny edge detection
    low_t = 70
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    
    # Apply Region of Interest Mask
    region = region_selection(edges)
    
    # Apply Hough Transform to get lane lines
    hough = hough_transform(region)
    
    # Draw the detected lane lines
    result = draw_lane_lines(image, lane_lines(image, hough))

    # Return intermediate frames for debugging
    return grayscale, blur, edges, region, hough, result

def yolo_detection_process(frame_queue, result_queue, pid_controller, serial_queue):
    
    prev_pid = 0

    model = YOLO("BFMCv13Test2_ncnn_model/BFMCv13Test2_ncnn_model", task='segment')
    target_cX = 160  
    prev_time = time.time()

    cont = 500

    while True:
        try:
            frame = frame_queue.get(timeout=1)  
        except:
            continue  

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #frame = cv2.bitwise_not(frame)


        #frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=0)


        results = model.predict(source=frame, show=False, save=False, conf=0.3, imgsz=320, verbose=False)
        use_pid = False
        annotated_frame = results[0].plot()

        cX = 160 
        for result in results:
            for c in result:
                label_index = c.boxes.cls[0]
                if label_index == 9:  
                    b_mask = np.zeros(annotated_frame.shape[:2], np.uint8)
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                    use_pid = True
                    M = cv2.moments(b_mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(annotated_frame, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, "Center", (cX - 10, cY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if label_index == 8 and cont >= 200:
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    serial_queue.put('#speed:0;;\r\n')
                    time.sleep(3)
                    serial_queue.put('#speed:100;;\r\n')
                    time.sleep(0.5)
                    cont = 0

        print(cont)
        cont+= 1

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        error = cX - target_cX 
        if use_pid == True:
            pid_output = int(pid_controller.update(error, dt))
            prev_pid = pid_output

        else:
            frame = cv2.resize(frame,(320,320))
            _,_,_,_,_,annotated_frame = frame_processor(frame)
            pid_output = prev_pid
        

        pid_output = np.clip(pid_output,-230,230)

        pid_text = f"PID Output: {pid_output}"
        cv2.putText(annotated_frame, pid_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 30), 2)

        # Send the PID value to the serial queue
        serial_queue.put('#steer:'+str(pid_output)+';;\r\n')

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
            
            # Get intermediate frames for debugging
            grayscale, blur, edges, region, hough, processed_frame = frame_processor(frame)

            # Concatenate frames horizontally to show the transformations
            
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


# Serial Reading Process (Non-blocking)
def read_serial_process(serial_queue):
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            serial_queue.put(data)
            print(data)
        time.sleep(0.01)  # Add a small delay to prevent CPU overload


# Serial Writing Process (Non-blocking)
def write_serial_process(serial_queue):
    while True:
        if not serial_queue.empty():
            value = serial_queue.get()
            print(value)
            #ser.write(f'#steer:{value};;\r\n'.encode('utf-8'))  # Write the PID value to the serial port
            ser.write(str(value).encode('utf-8'))
        time.sleep(0.01)  # Add a small delay to prevent CPU overload


if __name__ == "__main__":
    pid_controller = PIDController(Kp=5.5, Ki=0.00, Kd=0.0)

    # Queue for serial communication (moved before yolo_process)
    serial_queue = multiprocessing.Queue()

    ser.write('#kl:30;;\r\n'.encode('utf-8'))
    print("Done")
    time.sleep(3)  # Allow some time for the serial port to initialize
    ser.write('#steer:50;;\r\n'.encode('utf-8'))
    ser.write('#speed:100;;\r\n'.encode('utf-8'))


    # Start YOLO detection process
    yolo_process = multiprocessing.Process(target=yolo_detection_process, args=(frame_queue, result_queue, pid_controller, serial_queue))
    yolo_process.start()
    


    # Start serial read and write processes
    serial_read_process = multiprocessing.Process(target=read_serial_process, args=(serial_queue,))
    serial_write_process = multiprocessing.Process(target=write_serial_process, args=(serial_queue,))

    serial_read_process.start()
    serial_write_process.start()

    try:
        app.run(host='0.0.0.0', port=5050)
    except KeyboardInterrupt:
        print("Shutting down...")

    yolo_process.terminate()
    yolo_process.join()

    serial_read_process.terminate()
    serial_write_process.terminate()

    serial_read_process.join()
    serial_write_process.join()
