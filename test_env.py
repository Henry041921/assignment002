import numpy as np
import mediapipe as mp

print(f"NumPy Version: {np.__version__}")

try:
    mp_hands = mp.solutions.hands
    print("MediaPipe Loading successful!")
except Exception as e:
    print(f"It still doesn't work: {e}")