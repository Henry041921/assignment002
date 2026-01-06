# Group ID: B109
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import copy

DATA_DIR = "images"
OUTPUT_FILE = "landmarks_data.csv"

# --- Init MediaPipe ---
# Use static mode for images, only detect 1 hand.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def normalize_landmarks(landmarks):
    """
    Normalize data.
    Make coordinates relative to wrist and scale size to -1 to 1.
    """
    temp_landmark_list = copy.deepcopy(landmarks)

    # --- Step 1: Relative Coordinates ---
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            # Wrist (point 0) becomes the base point (0,0,0).
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        # All points minus the base point.
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # --- Step 2: Scale Normalization ---
    # Find max absolute value to scale everything.
    flattened = [val for sublist in temp_landmark_list for val in sublist]
    max_value = max(list(map(abs, flattened)))

    def normalize_(n):
        # Prevent divide by zero.
        return n / max_value if max_value != 0 else 0

    final_features = []
    for lm in temp_landmark_list:
        # Normalize x, y, z.
        final_features.extend([normalize_(lm[0]), normalize_(lm[1]), normalize_(lm[2])])

    return final_features

def process_dataset():
    data = []

    # Check if folder exists.
    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder '{DATA_DIR}' not found.")
        return

    # Get class names like A, B, C.
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Classes found: {classes}")

    total_images = 0
    valid_images = 0

    # Loop each class folder.
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        file_names = os.listdir(class_path)

        print(f"Processing class [{class_name}]...")

        for file_name in file_names:
            # Check image format.
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total_images += 1
            image_path = os.path.join(class_path, file_name)

            img = cv2.imread(image_path)
            if img is None:
                continue

            # MediaPipe needs RGB format.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Data Cleaning: Save only if hand detected.
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Get raw x, y, z.
                raw_landmarks = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.append([lm.x, lm.y, lm.z])

                # Do normalization.
                processed_features = normalize_landmarks(raw_landmarks)

                # Save label and features.
                row = [class_name] + processed_features
                data.append(row)
                valid_images += 1

    # Save to CSV file.
    if data:
        # Create header: label, x0, y0, z0...
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])

        df = pd.DataFrame(data, columns=header)
        df.to_csv(OUTPUT_FILE, index=False)

        print("-" * 40)
        print(f"🎉 Feature extraction and preprocessing completed!")
        print(f"original image: {total_images} 张")
        print(f"Valid data after cleaning: {valid_images} 条")
        print(f"Data has been saved to: {OUTPUT_FILE}")
        print("-" * 40)
    else:
        print("No data extracted, please check the image path.")

if __name__ == "__main__":
    process_dataset()
    hands.close()