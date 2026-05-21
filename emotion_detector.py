import cv2
from deepface import DeepFace
import pyttsx3
import threading
import time

# Text-to-speech
engine = pyttsx3.init()

# Webcam
cap = cv2.VideoCapture(0)

last_emotion = ""
display_emotion = "Detecting..."
last_detection_time = 0

# Delay between detections (seconds)
DETECTION_INTERVAL = 2

# Speak function
def speak(emotion):
    engine.say(f"You look {emotion}")
    engine.runAndWait()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    # Detect only every 2 seconds
    if current_time - last_detection_time > DETECTION_INTERVAL:

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']

            # Update only if emotion changes
            if emotion != last_emotion:
                last_emotion = emotion
                display_emotion = emotion

                # Voice output
                threading.Thread(
                    target=speak,
                    args=(emotion,),
                    daemon=True
                ).start()

            last_detection_time = current_time

        except Exception as e:
            print("Error:", e)

    # Show emotion on screen
    cv2.putText(
        frame,
        f'Emotion: {display_emotion}',
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Stable Emotion Detector", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()