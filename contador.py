import cv2
import mediapipe as mp
import numpy as np
import pygame 
# from tkinter import messagebox, Tk

# root = Tk()
# root.withdraw()

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("levantense.mp3")  

def playaudio():
    if not pygame.mixer.music.get_busy():  
        pygame.mixer.music.play(loops=1)

def stopaudio():
    pygame.mixer.music.stop()

def drawing_output(frame, coordinates_left_eye, coordinates_rigth_eye, blink_counter, microsleep_counter):
    aux_image = np.zeros(frame.shape, dtype=np.uint8)

    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_rigth_eye])
    cv2.rectangle(aux_image, (0, 0), (115, 50), (150, 0, 150), -1)
    cv2.rectangle(aux_image, (115, 0), (140, 50), (150, 0, 150), 2)

    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1) 

    cv2.rectangle(output, (0, 0), (115, 50), (150, 0, 150), -1)
    cv2.rectangle(output, (115, 0), (140, 50), (150, 0, 150), 2)

    cv2.putText(output, "parpadeos:", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(blink_counter), (120, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128, 0, 250), 2)
    cv2.putText(output, "micro sleep:", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(microsleep_counter), (120, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128, 0, 250), 2)

    return output

def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))  
    return (d_A + d_B) / (2 * d_C)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_rigth_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.26
NUM_FRAMES = 2
aux_counter = 0
blink_counter = 0
microsleep_counter = 0


fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
MICROSLEEP_FRAMES = int(fps * 3) 

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
        coordinates_rigth_eye = []

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    punto = face_landmarks.landmark[index]
                    x = int(punto.x * width)
                    y = int(punto.y * height)
                    coordinates_left_eye.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 255), 1)

                for index in index_rigth_eye:
                    punto = face_landmarks.landmark[index]
                    x = int(punto.x * width)
                    y = int(punto.y * height)
                    coordinates_rigth_eye.append((x, y))
                    cv2.circle(frame, (x, y), 2, (128, 0, 255), 1)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

            ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
            ear_rigth_eye = eye_aspect_ratio(coordinates_rigth_eye)
            ear = (ear_left_eye + ear_rigth_eye)/2

            if ear < EAR_THRESH:
                aux_counter += 1
                if aux_counter >= MICROSLEEP_FRAMES:
                    microsleep_counter += 1
                    cv2.putText(frame, "MICROSUEÑO", (150, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    playaudio() 
                    messagebox.showwarning("ALERTA", "YA TE DORMISTE")
            else:
                if aux_counter >= NUM_FRAMES and aux_counter < MICROSLEEP_FRAMES:
                    blink_counter += 1
                aux_counter = 0

            frame = drawing_output(frame, coordinates_left_eye, coordinates_rigth_eye, blink_counter, microsleep_counter)


        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  
            break

cap.release()
cv2.destroyAllWindows()
