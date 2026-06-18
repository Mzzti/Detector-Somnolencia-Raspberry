from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture("tcp://127.0.0.1:8888")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame")
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return '<img src="/video" style="width:800px; max-width:95%;">'

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000)
