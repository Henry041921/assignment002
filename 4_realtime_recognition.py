import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox, font

MODEL_FILE = "gesture_model.pkl"

# Load the trained model from the file.
print(f"Loading model from {MODEL_FILE}...")
try:
    classifier = joblib.load(MODEL_FILE)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    # If model is not found, show error and exit.
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Error", f"Model file {MODEL_FILE} not found\nPlease run 3_train_model.py first")
    exit()

# Initialize MediaPipe tools for hand detection.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(landmarks):
    # Create a copy to avoid modifying original data.
    temp_landmark_list = copy.deepcopy(landmarks)

    # Convert to relative coordinates based on the wrist point.
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Flatten the list to find the max absolute value.
    flattened = [val for sublist in temp_landmark_list for val in sublist]
    max_value = max(list(map(abs, flattened)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    # Normalize all features to be between -1 and 1.
    final_features = []
    for lm in temp_landmark_list:
        final_features.extend([normalize_(lm[0]), normalize_(lm[1]), normalize_(lm[2])])

    return final_features


def predict_and_draw(image, hands_module):
    # Convert BGR image to RGB for MediaPipe.
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands_module.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand connections on the frame.
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get raw coordinates from landmarks.
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.append([lm.x, lm.y, lm.z])

            # Preprocess data to match training format.
            processed_features = normalize_landmarks(raw_landmarks)

            input_data = np.array([processed_features])

            try:
                # Predict the gesture using the classifier.
                prediction = classifier.predict(input_data)
                predicted_label = prediction[0]

                # Show confidence score if available.
                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(input_data)
                    confidence = np.max(proba)
                    display_text = f"{predicted_label} ({confidence * 100:.1f}%)"
                else:
                    display_text = f"Gesture: {predicted_label}"

                # Draw a background box for better text visibility.
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                cv2.rectangle(image, (0, 0), (text_w + 20, text_h + 40), (0, 0, 0), -1)
                cv2.putText(image, display_text, (10, text_h + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            except Exception as e:
                print(f"Prediction error: {e}")

    return image


def run_camera_mode():
    print("\n🚀 Starting camera...")

    # Configure MediaPipe for stream input.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera!")
        return

    prev_frame_time = 0

    # Main loop for video processing.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect.
        frame = cv2.flip(frame, 1)

        frame = predict_and_draw(frame, hands)

        # Calculate and display FPS.
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, "Press 'Q' to Exit", (frame.shape[1] - 200, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('ASL Recognition (Camera Mode)', frame)

        # Exit loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Camera mode ended")


def run_image_mode():
    # Open file dialog to select an image.
    file_path = filedialog.askopenfilename(
        title="Select a gesture image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    if not file_path:
        print("Selection cancelled")
        return

    # Use higher confidence for static images.
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    frame = cv2.imread(file_path)
    if frame is None:
        messagebox.showerror("Error", "Cannot read image, please check the file.")
        hands.close()
        return

    print(f"Analyzing: {file_path} ...")

    frame = predict_and_draw(frame, hands)

    cv2.putText(frame, "Press Any Key to Close", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    window_name = f'Result: {os.path.basename(file_path)}'
    cv2.imshow(window_name, frame)

    # Wait for user input to close the window.
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hands.close()


def start_gui_app():
    # Set up the main GUI window.
    root = tk.Tk()
    root.title("ASL Gesture Recognition System")
    root.geometry("400x350")

    # Center the window on the screen.
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 400) // 2
    y = (screen_height - 350) // 2
    root.geometry(f"400x350+{x}+{y}")

    title_font = font.Font(family="Helvetica", size=16, weight="bold")
    btn_font = font.Font(family="Helvetica", size=12)

    lbl_title = tk.Label(root, text="ASL Gesture Recognition", font=title_font, pady=20)
    lbl_title.pack()

    lbl_desc = tk.Label(root, text="Please select mode:", font=("Arial", 10), fg="gray")
    lbl_desc.pack(pady=5)

    # Button to start the camera mode.
    btn_cam = tk.Button(root, text="Start Camera (Real-time)",
                        font=btn_font, bg="#e1f5fe", height=2, width=30,
                        command=run_camera_mode)
    btn_cam.pack(pady=10)

    # Button to upload an image file.
    btn_img = tk.Button(root, text="Upload Image (Select File)",
                        font=btn_font, bg="#fce4ec", height=2, width=30,
                        command=run_image_mode)
    btn_img.pack(pady=10)

    # Button to exit the application.
    btn_exit = tk.Button(root, text="Exit Program",
                         font=btn_font, height=1, width=30,
                         command=root.quit)
    btn_exit.pack(pady=20)


    root.mainloop()


if __name__ == "__main__":
    start_gui_app()