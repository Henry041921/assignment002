import numpy as np
import mediapipe as mp

print(f"NumPy Version: {np.__version__}")

try:
    # Python 3.10 下，这行代码应该直接能跑，不需要任何技巧
    mp_hands = mp.solutions.hands
    print("✅ MediaPipe 加载成功！(环境重做有效)")
except Exception as e:
    print(f"❌ 还是不行: {e}")