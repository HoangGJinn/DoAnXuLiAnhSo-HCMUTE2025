import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "Source" / "NhanDienTraiCay"
MODEL_PATH = MODEL_DIR / "trai_cay_yolov8n.onnx"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.3

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

CLASSES = ['SauRieng', 'Tao', 'ThanhLong','DuaHau','Cam']


def draw_label(im, label, x, y):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    rect_y = max(0, y - dim[1] - baseline)
    cv2.rectangle(im, (x, rect_y), (x + dim[0], y + baseline), BLACK, cv2.FILLED)
    cv2.putText(im, label, (x, y), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    try:
        blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs
    except Exception as e:
        st.error(f"Error during pre-processing or model inference: {e}")
        return None

def post_process(input_image, outputs):
    if outputs is None or not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
        st.warning("Model outputs are invalid or empty.")
        return input_image
    try:
        predictions = outputs[0]
        if predictions.shape[0] != 1:
            st.warning(f"Unexpected output batch size: {predictions.shape[0]}")
        num_detections = predictions.shape[2]
        # Số lớp = tổng số cột - 4 cột bbox
        num_classes_output = predictions.shape[1] - 4
        if num_classes_output != len(CLASSES):
            st.warning(
                f"Model output has {num_classes_output} classes, but script defines {len(CLASSES)} classes. Mismatch may occur."
            )
        output_reshaped = predictions[0].T

        boxes = []
        confidences = []  # lưu max_class_score của các box tiềm năng
        class_ids = []  # class_id tương ứng

        image_height, image_width = input_image.shape[:2]
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        # Bước 1: Lặp qua tất cả các dự đoán và lọc dựa trên điểm lớp cao nhất
        for i in range(num_detections):
            row = output_reshaped[i]
            # Lấy tọa độ bbox (4 cột đầu)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            # Lấy điểm của tất cả các lớp (từ cột 4 trở đi)
            classes_scores = row[4:]
            if classes_scores.size == 0:  # Kiểm tra nếu không có điểm lớp nào
                continue
            # Tìm lớp có điểm cao nhất và điểm số của nó
            class_id = np.argmax(classes_scores)
            max_class_score = classes_scores[class_id]

            # SCORE_THRESHOLD
            if max_class_score >= SCORE_THRESHOLD:
                # Lưu trữ nếu điểm đủ cao
                confidences.append(float(max_class_score))
                class_ids.append(class_id)
                # Tính toán và lưu trữ bounding box
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        # Bước 2: Non-Maximum Suppression
        # hoạt động trên các box đã vượt qua ngưỡng SCORE_THRESHOLD
        indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
        output_image = input_image.copy()
        if len(indices) > 0:
            if isinstance(indices[0], (list, np.ndarray)):
                indices = indices.flatten()

            for i in indices:  # chỉ số của box được giữ lại sau NMS
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                confidence = confidences[i]  # là max_class_score đã lưu
                retrieved_class_id = class_ids[i]  # class_id tương ứng

                # vẽ
                if 0 <= retrieved_class_id < len(CLASSES):
                    label = "{}:{:.2f}".format(CLASSES[retrieved_class_id], confidence)
                    color = BLUE
                else:
                    print(
                        f"--- WARNING (Post-NMS): Invalid class_id {retrieved_class_id} detected (max is {len(CLASSES) - 1}). ---")
                    label = "Unknown_ID_{}:{:.2f}".format(retrieved_class_id, confidence)
                    color = (0, 0, 255)

                cv2.rectangle(output_image, (left, top), (left + width, top + height), color, 2 * THICKNESS)
                draw_label(output_image, label, left, top - 5)

        return output_image

    except Exception as e:
        st.error(f"Error during post-processing: {e}")
        import traceback
        print(traceback.format_exc())
        return input_image

# Cached Model Loading
@st.cache_resource
def load_model(model_path: Path):
    model_path_str = str(model_path)
    if not model_path.is_file():
        st.error(f"Model file not found at: {model_path_str}")
        return None
    try:
        net = cv2.dnn.readNet(model_path_str)
        return net
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

st.title("Nhận diện 5 loại trái cây 🍈 🍎 🥭 🍉 🍊")
net = load_model(MODEL_PATH)
img_file_buffer = st.file_uploader("Tải lên hình ảnh trái cây", type=["bmp", "png", "jpg", "jpeg"], key="obj_uploader")
if img_file_buffer is not None and net is not None:
    try:
        image_pil = Image.open(img_file_buffer)
        # Convert to OpenCV format (NumPy array, BGR)
        frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        st.image(image_pil, caption="Ảnh đã tải lên")
        if st.button('Bắt đầu nhận diện', key='predict_obj'):
            with st.spinner("Đang xử lý..."):
                detections = pre_process(frame_bgr, net)
                img_result = post_process(frame_bgr.copy(), detections)
                try:
                    t, _ = net.getPerfProfile()
                    label_perf = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                    cv2.putText(img_result, label_perf, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
                    st.caption(label_perf)
                except Exception as e_perf:
                    st.warning(f"Could not get performance profile: {e_perf}")

                img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                st.image(img_result_rgb, caption="Kết quả nhận diện")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")

elif net is None:
    st.error("Không thể tải mô hình nhận diện. Vui lòng kiểm tra lại đường dẫn và file model.")

st.info("Tải lên một hình ảnh chứa các loại trái cây như Sầu Riêng, Táo, Thanh Long, Dưa Hấu, Cam để bắt đầu.")
