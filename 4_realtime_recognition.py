import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox, font

# --- 配置 ---
MODEL_FILE = "gesture_model.pkl"

# 1. 加载训练好的模型
print(f"Loading model from {MODEL_FILE}...")
try:
    classifier = joblib.load(MODEL_FILE)
    print("✅ 模型加载成功！")
except FileNotFoundError:
    # 如果没有模型，弹窗提示并退出
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("错误", f"找不到模型文件 {MODEL_FILE}\n请先运行 3_train_model.py")
    exit()

# 初始化 MediaPipe 绘图工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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


def predict_and_draw(image, hands_module):
    """
    通用处理函数：接收一张图片（或视频帧），进行检测、预测并绘制结果。
    """
    # 转为 RGB 供 MediaPipe 使用
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 检测手势
    results = hands_module.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # A. 画骨架
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # B. 提取原始坐标
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.append([lm.x, lm.y, lm.z])

            # C. 🔥 执行归一化
            processed_features = normalize_landmarks(raw_landmarks)

            # D. AI 预测
            input_data = np.array([processed_features])

            try:
                prediction = classifier.predict(input_data)
                predicted_label = prediction[0]

                # 获取置信度
                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(input_data)
                    confidence = np.max(proba)
                    display_text = f"{predicted_label} ({confidence * 100:.1f}%)"
                else:
                    display_text = f"Gesture: {predicted_label}"

                # E. 在屏幕上显示结果
                # 获取文字大小以便动态调整背景框
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                cv2.rectangle(image, (0, 0), (text_w + 20, text_h + 40), (0, 0, 0), -1)
                cv2.putText(image, display_text, (10, text_h + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            except Exception as e:
                print(f"预测出错: {e}")

    return image


def run_camera_mode():
    """模式 1: 实时摄像头识别"""
    print("\n🚀 正在启动摄像头...")

    # 视频模式下 static_image_mode=False 更快
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        return

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像翻转
        frame = cv2.flip(frame, 1)

        # 核心处理
        frame = predict_and_draw(frame, hands)

        # 显示 FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 显示提示信息
        cv2.putText(frame, "Press 'Q' to Exit", (frame.shape[1] - 200, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('ASL Recognition (Camera Mode)', frame)

        # 按 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("🎥 摄像头模式已结束")


def run_image_mode():
    """模式 2: 单张图片识别 (文件选择器)"""

    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择一张手势图片",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    if not file_path:
        print("取消选择")
        return

    # 图片模式下 static_image_mode=True 精度更高
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    frame = cv2.imread(file_path)
    if frame is None:
        messagebox.showerror("错误", "无法读取图片，请确保文件未损坏。")
        hands.close()
        return

    print(f"🖼️ 正在分析: {file_path} ...")

    # 核心处理
    frame = predict_and_draw(frame, hands)

    # 显示提示
    cv2.putText(frame, "Press Any Key to Close", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示结果
    window_name = f'Result: {os.path.basename(file_path)}'
    cv2.imshow(window_name, frame)

    # 等待任意键关闭
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hands.close()


# ==========================================
# 🖥️ GUI 主界面逻辑
# ==========================================
def start_gui_app():
    # 创建主窗口
    root = tk.Tk()
    root.title("ASL 手势识别系统")
    root.geometry("400x350")

    # 设置居中
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 400) // 2
    y = (screen_height - 350) // 2
    root.geometry(f"400x350+{x}+{y}")

    # 字体设置
    title_font = font.Font(family="Helvetica", size=16, weight="bold")
    btn_font = font.Font(family="Helvetica", size=12)

    # 标题标签
    lbl_title = tk.Label(root, text="🖐️ ASL Gesture Recognition", font=title_font, pady=20)
    lbl_title.pack()

    # 说明标签
    lbl_desc = tk.Label(root, text="请选择识别模式：", font=("Arial", 10), fg="gray")
    lbl_desc.pack(pady=5)

    # --- 按钮区域 ---
    # 摄像头按钮
    btn_cam = tk.Button(root, text="📹 启动摄像头 (Real-time)",
                        font=btn_font, bg="#e1f5fe", height=2, width=30,
                        command=run_camera_mode)  # 点击调用 run_camera_mode
    btn_cam.pack(pady=10)

    # 图片按钮
    btn_img = tk.Button(root, text="🖼️ 上传图片识别 (Upload Image)",
                        font=btn_font, bg="#fce4ec", height=2, width=30,
                        command=run_image_mode)  # 点击调用 run_image_mode
    btn_img.pack(pady=10)

    # 退出按钮
    btn_exit = tk.Button(root, text="❌ 退出程序 (Exit)",
                         font=btn_font, height=1, width=30,
                         command=root.quit)
    btn_exit.pack(pady=20)

    # 底部版权
    lbl_footer = tk.Label(root, text="Powered by MediaPipe & Scikit-Learn", font=("Arial", 8), fg="#ccc")
    lbl_footer.pack(side=tk.BOTTOM, pady=5)

    # 启动 GUI 循环
    root.mainloop()


if __name__ == "__main__":
    start_gui_app()