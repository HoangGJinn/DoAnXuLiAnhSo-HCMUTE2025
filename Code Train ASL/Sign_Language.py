import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3


# Load model và encoder
model = tf.keras.models.load_model('ASL_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Prediction history
prediction_history = deque(maxlen=10)
predicted_word = ""

# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("ASL Sign Language Recognition")
window.geometry("1280x600")  # Tăng chiều cao để có chỗ cho khu vực chữ

# --- Bố cục giao diện ---
# Khung trên chứa webcam
frame_top = tk.Frame(window)
frame_top.pack()

# Nhãn hiển thị video
video_label = tk.Label(frame_top)
video_label.pack()

# Nhãn hiển thị từ đã nhận diện
word_label = tk.Label(window, text="Word: ", font=("Helvetica", 24), fg="blue")
word_label.pack(pady=10, anchor='w', padx=20)

# Khởi tạo engine TTS
engine = pyttsx3.init()

def speak_word():
    word = predicted_word  # Hoặc: word_label.cget("text").replace("Word: ", "")
    if word:
        engine.say(word)
        engine.runAndWait()

# Khung chứa các nút
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

# Nút đọc
read_button = tk.Button(button_frame, text="🔊 Đọc từ", font=("Helvetica", 16), command=speak_word)
read_button.pack(side=tk.LEFT, padx=10)

# Nút reset từ
def reset_word():
    global predicted_word
    predicted_word = ""
    word_label.config(text="Word: ")

reset_button = tk.Button(button_frame, text="Reset Word", font=("Helvetica", 14), command=reset_word)
reset_button.pack(side=tk.LEFT, padx=10)

# Webcam
cap = cv2.VideoCapture(0)

check = True
def update_frame():
    global check, predicted_word

    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Nền trắng để vẽ tay
    hand_only_canvas = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
    current_prediction = ""
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints_np = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints_np, verbose=0)
            confidence = np.max(prediction)
            predicted_letter = le.inverse_transform([np.argmax(prediction)])[0]
            current_prediction = predicted_letter

            if predicted_letter == "next":
                check = True
            else:
                if confidence > 0.7:
                    prediction_history.append(predicted_letter)

            for connection in mp_hands.HAND_CONNECTIONS:
                start = hand_landmarks.landmark[connection[0]]
                end = hand_landmarks.landmark[connection[1]]
                x_start = int(start.x * frame.shape[1])
                y_start = int(start.y * frame.shape[0])
                x_end = int(end.x * frame.shape[1])
                y_end = int(end.y * frame.shape[0])
                cv2.line(hand_only_canvas, (x_start, y_start), (x_end, y_end), (0, 0, 0), 2)
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(hand_only_canvas, (x, y), 5, (0, 0, 255), -1)

    if (len(set(prediction_history)) == 1
            and len(prediction_history) == prediction_history.maxlen
            and current_prediction not in ["next"]):
        if current_prediction == "del":
            predicted_word = predicted_word[:-1]
            prediction_history.clear()
        elif check:
            if current_prediction == "space":
                predicted_word += " "
            else:
                predicted_word += current_prediction
            check = False
            prediction_history.clear()

    cv2.putText(frame, f"Letter: {current_prediction} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Check: {check}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Ghép hai ảnh cạnh nhau
    hand_only_canvas_resized = cv2.resize(hand_only_canvas, (frame.shape[1], frame.shape[0]))
    combined_frame = np.hstack((frame, hand_only_canvas_resized))

    rgb_image = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Cập nhật chữ hiển thị phía dưới webcam
    word_label.config(text=f"Word: {predicted_word}")

    window.after(10, update_frame)

def on_closing():
    cap.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
window.mainloop()
