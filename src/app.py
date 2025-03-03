from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("best.pt")  # Ensure 'best.pt' is in your project folder

# Initialize Video Capture (Webcam)
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 helmet detection
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = "Helmet" if box.conf[0] > 0.5 else "No Helmet"
                color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route for video feed
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
