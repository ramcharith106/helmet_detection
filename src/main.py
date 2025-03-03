import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cvzone
import easyocr
import os
from datetime import datetime
import xlwings as xw

app = Flask(__name__)

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Load YOLOv8 model
model = YOLO("best.pt")
names = model.names

# Define polygon area
area = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]

# Create directory for current date
current_date = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(current_date):
    os.makedirs(current_date)

# Initialize Excel file path
excel_file_path = os.path.join(current_date, f"{current_date}.xlsx")
wb = xw.Book(excel_file_path) if os.path.exists(excel_file_path) else xw.Book()
ws = wb.sheets[0]
if ws.range("A1").value is None:
    ws.range("A1").value = ["Number Plate", "Date", "Time"]

processed_track_ids = set()
cap = cv2.VideoCapture("final.mp4")  # Read from video file

def perform_ocr(image_array):
    if image_array is None:
        return ""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    
    detected_text = "".join([res[1] for res in results])
    return detected_text

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        no_helmet_detected = False
        numberplate_box = None
        numberplate_track_id = None

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:
                    if c == 'no-helmet':
                        no_helmet_detected = True
                    elif c == 'numberplate':
                        numberplate_box = box
                        numberplate_track_id = track_id
            
            if no_helmet_detected and numberplate_box is not None and numberplate_track_id not in processed_track_ids:
                x1, y1, x2, y2 = numberplate_box
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (120, 85))

                text = perform_ocr(crop)
                
                current_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
                crop_image_path = os.path.join(current_date, f"{text}_{current_time}.jpg")
                cv2.imwrite(crop_image_path, crop)

                last_row = ws.range("A" + str(ws.cells.last_cell.row)).end('up').row
                ws.range(f"A{last_row+1}").value = [text, current_date, current_time]

                processed_track_ids.add(numberplate_track_id)

        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_data', methods=['GET'])
def get_detected_data():
    data = []
    for row in ws.range("A2:C" + str(ws.cells.last_cell.row)).value:
        if row:
            data.append({"number_plate": row[0], "date": row[1], "time": row[2]})
    
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
