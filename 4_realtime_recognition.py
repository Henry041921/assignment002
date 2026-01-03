import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import copy

# --- 配置 ---
MODEL_FILE = "gesture_model.pkl"

# 1. 加载训练好的模型
print(f"Loading model from {MODEL_FILE}...")
try:
    classifier = joblib.load(MODEL_FILE)
    print("✅ 模型加载成功！")
except FileNotFoundError:
    print("❌ 错误: 找不到模型文件。请先运行 3_train_model.py")
    exit()

# 2. 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,  # 视频流模式
    max_num_hands=1,  # 只识别一只手
    min_detection_confidence=0.7,  # 提高一点门槛，防止乱飘
    min_tracking_confidence=0.5
)


def normalize_landmarks(landmarks):
    """
    🌟 核心关键：必须与训练时的预处理逻辑完全一致！
    """
    temp_landmark_list = copy.deepcopy(landmarks)

    # --- 1. 相对坐标转换 ---
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # --- 2. 尺度归一化 ---
    flattened = [val for sublist in temp_landmark_list for val in sublist]
    max_value = max(list(map(abs, flattened)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    final_features = []
    for lm in temp_landmark_list:
        final_features.extend([normalize_(lm[0]), normalize_(lm[1]), normalize_(lm[2])])

    return final_features


# 3. 打开摄像头
cap = cv2.VideoCapture(0)  # 如果外接摄像头，尝试改成 1
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("\n🎥 摄像头已启动！(按 'Q' 键退出)")

# FPS 计算变量
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 镜像翻转 (让你看着更自然)
    frame = cv2.flip(frame, 1)

    # 转为 RGB 供 MediaPipe 使用
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测手势
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # A. 画骨架
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # B. 提取原始坐标
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.append([lm.x, lm.y, lm.z])

            # C. 🔥 执行归一化 (关键一步！)
            processed_features = normalize_landmarks(raw_landmarks)

            # D. AI 预测
            # 将 list 转为 numpy 数组 (形状 1x63)
            input_data = np.array([processed_features])

            try:
                prediction = classifier.predict(input_data)
                predicted_label = prediction[0]

                # 获取置信度 (如果是 RF 或 KNN)
                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(input_data)
                    confidence = np.max(proba)
                    display_text = f"{predicted_label} ({confidence * 100:.1f}%)"
                else:
                    display_text = f"Gesture: {predicted_label}"

                # E. 在屏幕上显示结果
                cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)  # 黑色背景条
                cv2.putText(frame, display_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            except Exception as e:
                print(f"预测出错: {e}")

    # 显示 FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 显示画面
    cv2.imshow('ASL Recognition (High Accuracy Mode)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()