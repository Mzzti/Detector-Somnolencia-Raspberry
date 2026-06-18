from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import pygame

app = Flask(__name__)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("levantense.mp3")

def playaudio():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(loops=1)

def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter, microsleep_counter):
    cv2.rectangle(frame, (0, 0), (170, 60), (150, 0, 150), -1)
    cv2.putText(frame, f"Parpadeos: {blink_counter}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Microsuenos: {microsleep_counter}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return frame

def generate_frames():
    cap = cv2.VideoCapture("tcp://127.0.0.1:8888")

    mp_face_mesh = mp.solutions.face_mesh
    index_left_eye = [33, 160, 158, 133, 153, 144]
    index_right_eye = [362, 385, 387, 263, 373, 380]

    EAR_THRESH = 0.26
    NUM_FRAMES = 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    MICROSLEEP_FRAMES = int(fps * 3)

    aux_counter = 0
    blink_counter = 0
    microsleep_counter = 0

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            coordinates_left_eye = []
            coordinates_right_eye = []

            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    for index in index_left_eye:
                        point = face_landmarks.landmark[index]
                        x = int(point.x * width)
                        y = int(point.y * height)
                        coordinates_left_eye.append((x, y))
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)

                    for index in index_right_eye:
                        point = face_landmarks.landmark[index]
                        x = int(point.x * width)
                        y = int(point.y * height)
                        coordinates_right_eye.append((x, y))
                        cv2.circle(frame, (x, y), 2, (128, 0, 255), 1)

                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
                ear = (ear_left_eye + ear_right_eye) / 2

                if ear < EAR_THRESH:
                    aux_counter += 1
                    if aux_counter >= MICROSLEEP_FRAMES:
                        microsleep_counter += 1
                        cv2.putText(frame, "MICROSUEÑO", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        playaudio()
                else:
                    if NUM_FRAMES <= aux_counter < MICROSLEEP_FRAMES:
                        blink_counter += 1
                    aux_counter = 0

                frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter, microsleep_counter)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return """
    <html>
        <head>
            <title>Detector de Somnolencia</title>
        </head>
        <body style="text-align:center; font-family:Arial;">
            <h1>Detector de Somnolencia</h1>
            <p>Video en vivo desde Raspberry Pi</p>
            <img src="/video"
     style="
        width:800px;
        max-width:95%;
        border-radius:15px;
        border:3px solid #7b2cbf;
     ">
        </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000)
