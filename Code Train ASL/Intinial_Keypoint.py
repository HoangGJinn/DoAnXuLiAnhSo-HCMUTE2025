import cv2
import mediapipe as mp
import os
import csv

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Đường dẫn folder chính chứa ảnh theo từng ký tự
dataset_path = r'D:\DoAn_XuLiAnhSo\ASL_Alphabet_Dataset\asl_alphabet_train'
output_csv = 'keypoints_dataset.csv'

# Hàm kiểm tra ảnh có bị mờ không (dưới ngưỡng là mờ)
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold

# Hàm tăng tương phản bằng CLAHE
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Ghi dữ liệu ra CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)

    # Viết header
    header = []
    for i in range(21):  # 21 keypoints
        header += [f'x{i}', f'y{i}', f'z{i}']
    header += ['label']
    writer.writerow(header)

    # Lặp qua từng class (A, B, C, ...)
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            print(f"Không tìm thấy thư mục {class_dir}")
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Không thể đọc ảnh {img_name}")
                continue

            # Resize ảnh về cùng kích thước
            image = cv2.resize(image, (400, 400))

            # Tăng tương phản
            image = enhance_contrast(image)

            # Kiểm tra ảnh mờ
            if is_blurry(image):
                print(f"Ảnh {img_name} bị mờ (có thể bỏ qua hoặc xử lý thêm)")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                print(f"✅ Phát hiện bàn tay trong ảnh {img_name}")
                hand = result.multi_hand_landmarks[0]
                keypoints = []
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])  # Normalize tự động 0-1
                keypoints.append(label)
                writer.writerow(keypoints)
            else:
                print(f"❌ Không phát hiện bàn tay trong ảnh {img_name}")
