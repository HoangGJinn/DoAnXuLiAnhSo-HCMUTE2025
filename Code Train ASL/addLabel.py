import cv2
import mediapipe as mp
import pandas as pd
import os
#    Đây là code để thêm label vào file .csv đã có sẵn

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Đường dẫn tới thư mục chứa ảnh mới
image_dir = 'hand_sign_dataset'
output_csv = 'keypoints_dataset.csv'

# Danh sách lưu data mới
data_rows = []

# Lặp qua từng ảnh
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        label = filename.split('_')[0]  # Lấy nhãn từ tên ảnh
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)
            data_rows.append(row)
        else:
            print(f"❌ Không phát hiện tay trong ảnh: {filename}")

# Tạo DataFrame
columns = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
df_new = pd.DataFrame(data_rows, columns=columns)

# Nếu file CSV đã tồn tại, thêm vào file cũ
if os.path.exists(output_csv):
    df_old = pd.read_csv(output_csv)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_csv(output_csv, index=False)
else:
    df_new.to_csv(output_csv, index=False)

print("✅ Đã thêm dữ liệu keypoints mới vào CSV!")
