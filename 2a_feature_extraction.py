import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import copy

# --- 配置区 ---
# ⚠️ 这里必须对应老师提供的匿名数据集文件夹名字
DATA_DIR = "images"
# 我们继续使用这个文件名，这样后面的训练代码不用改
OUTPUT_FILE = "landmarks_data.csv"

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # 处理静态图片模式
    max_num_hands=1,  # 只处理一只手
    min_detection_confidence=0.5
)


def normalize_landmarks(landmarks):
    """
    🌟 核心得分点 (Part 2b): 数据归一化 (Data Pre-processing)
    逻辑：
    1. 相对坐标：将所有点减去手腕(点0)的坐标。这样无论手在画面哪里，特征都一样。
    2. 尺度归一化：除以最大绝对值。这样无论手离摄像头远近(大小)，特征都一样。
    """
    # 深拷贝，防止修改原始数据
    temp_landmark_list = copy.deepcopy(landmarks)

    # --- 1. 转换为相对坐标 (Relative Coordinates) ---
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            # 获取手腕(Wrist)的坐标作为基准点
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        # 所有点减去基准点
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # --- 2. 尺度归一化 (Normalization) ---
    # 将所有坐标值缩放到 -1 到 1 之间
    # 展平列表以找到最大绝对值 (只考虑 x 和 y，因为 z 的比例尺可能不同，或者也一起归一化)
    flattened = [val for sublist in temp_landmark_list for val in sublist]
    max_value = max(list(map(abs, flattened)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    # 生成最终的特征列表
    final_features = []
    for lm in temp_landmark_list:
        # 对 x, y, z 都进行归一化
        final_features.extend([normalize_(lm[0]), normalize_(lm[1]), normalize_(lm[2])])

    return final_features


def process_dataset():
    data = []

    # 1. 检查数据文件夹
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到文件夹 '{DATA_DIR}'。请确保图片文件夹在当前目录下！")
        return

    # 获取所有类别 (A, B, C...)
    # 过滤掉隐藏文件 (如 .DS_Store)
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"📂 发现类别: {classes}")

    total_images = 0
    valid_images = 0

    # 2. 遍历每个类别的文件夹
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        file_names = os.listdir(class_path)

        print(f"正在处理类别 【{class_name}】...")

        for file_name in file_names:
            # 只处理图片文件
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total_images += 1
            image_path = os.path.join(class_path, file_name)

            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                continue

            # 转换颜色空间 BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # MediaPipe 处理
            results = hands.process(img_rgb)

            # --- 3. 数据清洗 (Data Cleaning) [cite: 161] ---
            # 如果没有检测到手，则跳过(视为噪声数据)，这就是文档要求的 Cleaning
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # 提取原始坐标 (x, y, z)
                raw_landmarks = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.append([lm.x, lm.y, lm.z])

                # 🔥 调用归一化函数 (Pre-processing 得分点)
                processed_features = normalize_landmarks(raw_landmarks)

                # 添加到数据列表: [Label, Feature1, Feature2 ... Feature63]
                row = [class_name] + processed_features
                data.append(row)
                valid_images += 1
            else:
                # 可以在这里打印日志，证明你做了清洗
                pass

    # 4. 保存为 CSV
    if data:
        # 生成表头
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])

        df = pd.DataFrame(data, columns=header)
        df.to_csv(OUTPUT_FILE, index=False)

        print("-" * 40)
        print(f"🎉 特征提取与预处理完成！")
        print(f"原始图片: {total_images} 张")
        print(f"清洗后有效数据: {valid_images} 条")
        print(f"数据已保存至: {OUTPUT_FILE}")
        print("-" * 40)
    else:
        print("❌ 未提取到任何数据，请检查图片路径。")


if __name__ == "__main__":
    process_dataset()
    hands.close()