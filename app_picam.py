from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (640, 360), "format": "RGB888"}
))
picam2.start()
time.sleep(2)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return """
    <html>
    <body style="text-align:center; font-family:Arial;">
        <h1>Camara Raspberry</h1>
        <img src="/video" style="width:800px; max-width:95%;">
    </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000)
