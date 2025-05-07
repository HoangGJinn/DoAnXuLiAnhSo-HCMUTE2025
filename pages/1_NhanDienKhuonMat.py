
import streamlit as st
import numpy as np
import cv2 as cv
import joblib
from pathlib import Path
import time
from PIL import Image


FACE_MODEL_DIR = Path(__file__).parent / 'Source' / 'KhuonMat'
SVM_MODEL_PATH = FACE_MODEL_DIR / 'svc.pkl'
DETECTOR_MODEL_PATH = FACE_MODEL_DIR / 'face_detection_yunet_2023mar.onnx'
RECOGNIZER_MODEL_PATH = FACE_MODEL_DIR / 'face_recognition_sface_2021dec.onnx'

MY_DICT = ['Duy cute','Giap Xo', 'Hung Rom' ,'Tai gay', 'Vinh 30cm']
COLORS = [(0, 255, 0), (255, 200, 200), (220, 220, 255), (255, 255, 0), (0, 255, 255)]

@st.cache_resource
def load_svm_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"SVM model not found at {path}")
        return None
    except Exception as e:
        st.error(f"Error loading SVM model: {e}")
        return None

@st.cache_resource
def load_face_detector(model_path, input_size=(320, 320), score_threshold=0.9, nms_threshold=0.3, top_k=5000):
    model_path_str = str(model_path)
    if not Path(model_path_str).is_file():
         st.error(f"Face Detector model file not found at {model_path_str}")
         return None
    try:
        detector = cv.FaceDetectorYN.create(
            model_path_str, "", input_size, score_threshold, nms_threshold, top_k
        )
        return detector
    except Exception as e:
        st.error(f"Error loading Face Detector model from {model_path_str}: {e}")
        return None

@st.cache_resource
def load_face_recognizer(model_path):
    model_path_str = str(model_path)
    if not Path(model_path_str).is_file():
         st.error(f"Face Recognizer model file not found at {model_path_str}")
         return None
    try:
        recognizer = cv.FaceRecognizerSF.create(model_path_str, "")
        return recognizer
    except Exception as e:
        st.error(f"Error loading Face Recognizer model from {model_path_str}: {e}")
        return None

def process_video_stream(video_source):
    st.info(f"Bắt đầu process_video_stream với nguồn: {video_source}")

    # --- Load Models ---
    # Tải trước khi tạo UI để đảm bảo chúng sẵn sàng
    # Các hàm cache sẽ trả về ngay nếu đã tải rồi
    svc = load_svm_model(SVM_MODEL_PATH)
    detector = load_face_detector(DETECTOR_MODEL_PATH)
    recognizer = load_face_recognizer(RECOGNIZER_MODEL_PATH)

    # Kiểm tra xem tất cả model/resource đã tải thành công chưa
    if not all([svc, detector, recognizer]):
        st.error("Không thể tải tất cả models/resources cần thiết. Vui lòng kiểm tra lại đường dẫn và file.")
        st.stop()

    st.success("Tất cả models và resources đã sẵn sàng.")

    # --- Tạo UI sau khi chắc chắn model đã có ---
    frame_window = st.image([])
    stop_button = st.button('Stop', key=f'stop_{video_source}')

    # Initialize stop state if not present
    # Sử dụng key cụ thể để tránh xung đột nếu hàm này được gọi nhiều lần với nguồn khác nhau
    session_key_stop = f'stop_processing_{video_source}'
    if session_key_stop not in st.session_state:
        st.session_state[session_key_stop] = False

    if stop_button:
        st.session_state[session_key_stop] = True
        st.info("Đã nhấn nút Stop.")

    if st.session_state[session_key_stop]:
        st.warning("Đang ở trạng thái dừng.")
    else:
         st.info("Trạng thái: Đang chạy.")

    # --- Mở Video Capture ---
    st.info(f"Đang mở VideoCapture cho nguồn: {video_source}...")
    cap = cv.VideoCapture(video_source)
    time.sleep(1) # Chờ một chút để camera/file có thời gian khởi tạo

    if not cap.isOpened():
        st.error(f"!!! Không thể mở nguồn video: {video_source}. Hãy kiểm tra camera hoặc đường dẫn file.")
        st.stop() # Dừng hẳn nếu không mở được video
    else:
        st.success(f"Đã mở thành công nguồn video: {video_source}")

    # Input size của detector
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Kiểm tra kích thước frame hợp lệ
    if frame_width <= 0 or frame_height <= 0:
        st.error(f"Kích thước video không hợp lệ: {frame_width}x{frame_height}. Kiểm tra lại nguồn video.")
        cap.release()
        st.stop()
    try:
         detector.setInputSize([frame_width, frame_height])
         st.info(f"Đã đặt InputSize cho detector: {frame_width}x{frame_height}")
    except Exception as e:
         st.error(f"Lỗi khi đặt InputSize cho detector: {e}")
         cap.release()
         st.stop()


    tm = cv.TickMeter()
    frame_count = 0 # Đếm số frame đã đọc

    # --- Main Loop ---
    while cap.isOpened():
        # Kiểm tra trạng thái dừng ngay đầu vòng lặp
        if st.session_state.get(session_key_stop, False):
            st.info("Phát hiện trạng thái dừng trong vòng lặp.")
            break # Thoát vòng lặp

        has_frame, frame = cap.read()
        frame_count += 1

        if not has_frame:
            st.warning(f"Không thể đọc thêm khung hình (đã đọc {frame_count} frames). Kết thúc luồng.")
            st.session_state[session_key_stop] = True # Đặt trạng thái dừng khi hết video
            break

        if frame is None or frame.size == 0:
             st.warning(f"Đọc được frame nhưng frame trống ở frame thứ {frame_count}.")
             continue # Bỏ qua frame lỗi này

        # --- Chỉ xử lý và hiển thị nếu không ở trạng thái dừng ---
        if not st.session_state[session_key_stop]:
            # Copy frame để vẽ
            display_frame = frame.copy()

            tm.start()
            faces = detector.detect(display_frame)
            tm.stop()

            # --- Process Detected Faces ---
            if faces[1] is not None:
                 for idx,face in enumerate(faces[1][:5]):
                     try:
                         face_align = recognizer.alignCrop(frame, face)  # Dùng frame gốc để align
                         face_feature = recognizer.feature(face_align)

                         test_predict = svc.predict(face_feature)
                         result = "Error"
                         color = (0, 0, 0)
                         confidence_threshold = 0.1  # Ngưỡng tin cậy

                         if hasattr(svc, "decision_function"):
                             try:
                                 # Tính điểm tin cậy lớn nhất
                                 confidence_score = np.max(svc.decision_function(face_feature))

                                 if confidence_score < confidence_threshold:
                                     result = "Cannot Recognize!"
                                     color = (128, 128, 128)  # Màu xám
                                 # Nếu đủ tin cậy, kiểm tra index và lấy tên/màu
                                 elif 0 <= test_predict[0] < len(MY_DICT):
                                     result = MY_DICT[test_predict[0]]
                                     color = COLORS[test_predict[0]]
                                 else:
                                     result = "Unknown Index"
                                     color = (0, 0, 255)  # Màu đỏ cho lỗi index
                             except Exception as e_conf:
                                 print(f"Error calculating/using confidence score: {e_conf}")
                                 result = "Confidence Error"
                                 color = (0, 0, 255)  # Màu đỏ

                         else:  # Trường hợp model không có decision_function
                             st.warning("SVM model does not have decision_function. Using index check only.")
                             if 0 <= test_predict[0] < len(MY_DICT):
                                 result = MY_DICT[test_predict[0]]
                                 color = COLORS[test_predict[0]]
                             else:
                                 result = "Unknown"
                                 color = (0, 0, 0)  # unknown thì đen

                         coords = face[:4].astype(np.int32)
                         x1, y1, w, h = coords[0], coords[1], coords[2], coords[3]
                         x2, y2 = x1 + w, y1 + h
                         cv.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                         text_y = y1 - 10 if y1 > 10 else y1 + 15

                         # Get text size.
                         text_size = cv.getTextSize(result, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                         dim, baseline = text_size[0], text_size[1]
                         cv.rectangle(display_frame, (x1 - 5, text_y + 10), (x1 + dim[0], text_y + baseline - 35), color,
                                      cv.FILLED)

                         cv.putText(display_frame, result, (x1, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

                     except Exception as e:
                         pass

            # Draw FPS
            cv.putText(display_frame, f'FPS: {tm.getFPS():.2f}', (5, 20), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 3)

            frame_window.image(display_frame, channels='BGR')

    # --- Cleanup ---
    st.info("Kết thúc vòng lặp xử lý.")
    if cap.isOpened():
        cap.release()
        st.info("Đã giải phóng VideoCapture.")
    # Reset stop state for next interaction
    if session_key_stop in st.session_state:
        st.session_state[session_key_stop] = False
        st.info("Đã reset trạng thái dừng.")

def recognize_faces_in_image(image_bgr, detector, recognizer, svc):
    """Detects and recognizes faces in a static image."""
    st.info("Bắt đầu recognize_faces_in_image...")
    if not all([detector, recognizer, svc]):
        st.error("Models chưa được tải cho xử lý ảnh.")
        return image_bgr

    output_image = image_bgr.copy()
    height, width, _ = output_image.shape

    try:
        # Set input size for detector
        detector.setInputSize([width, height])
        st.info(f"Đặt InputSize cho detector (ảnh): {width}x{height}")
        faces = detector.detect(image_bgr)
        st.info(f"Kết quả detect: {type(faces)}") # In ra kiểu dữ liệu của faces
        if isinstance(faces, tuple) and len(faces) > 1 and faces[1] is not None:
            num_detected = len(faces[1])
            num_to_process = min(num_detected, 5)  # Lấy tối đa 5
            st.success(f"Phát hiện {num_detected} khuôn mặt. Đang xử lý {num_to_process} khuôn mặt.")

            for face_info in faces[1][:num_to_process]:  # Chỉ lặp qua tối đa 5 khuôn mặt đầu tiên
                 try:
                     # Align, Feature, Predict (giống process_video_stream)
                     face_align = recognizer.alignCrop(image_bgr, face_info)
                     face_feature = recognizer.feature(face_align)
                     test_predict = svc.predict(face_feature)
                     # --- Áp dụng logic confidence ---
                     result = "Error"
                     color = (0, 0, 0)
                     confidence_threshold = 0.1
                     if hasattr(svc, "decision_function"):
                         try:
                             confidence_score = np.max(svc.decision_function(face_feature))
                             if confidence_score < confidence_threshold:
                                 result = "Cannot Recognize!"
                                 color = (128, 128, 128)
                             elif 0 <= test_predict[0] < len(MY_DICT):
                                 result = MY_DICT[test_predict[0]]
                                 color = COLORS[test_predict[0] % len(COLORS)]
                             else:
                                 result = "Unknown Index"
                                 color = (0, 0, 255)
                         except Exception as e_conf:
                             result = "Confidence Error"
                             color = (0, 0, 255)
                     else:
                         if 0 <= test_predict[0] < len(MY_DICT):
                             result = MY_DICT[test_predict[0]]
                             color = COLORS[test_predict[0] % len(COLORS)]
                         else:
                             result = "Unknown"
                             color = (0, 0, 0)
                     # --- Kết thúc logic confidence ---

                     coords = face_info[:4].astype(np.int32)
                     x1, y1, w, h = coords[0], coords[1], coords[2], coords[3]
                     cv.rectangle(output_image, (x1, y1), (x1+w, y1+h), color, 3)
                     text_y = y1 - 10 if y1 > 10 else y1 + 15

                     # Get text size.
                     text_size = cv.getTextSize(result, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                     dim, baseline = text_size[0], text_size[1]
                     cv.rectangle(output_image, (x1 - 5, text_y+10), (x1 + dim[0], text_y + baseline - 35), color, cv.FILLED)
                     # Display text inside the rectangle.
                     cv.putText(output_image, result, (x1, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv.LINE_AA)

                     #cv.putText(output_image, result, (x1, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3, cv.LINE_AA)
                 except Exception as e_recog_img:
                     st.warning(f"Lỗi khi xử lý một khuôn mặt trong ảnh: {e_recog_img}")
                     try:
                          coords = face_info[:4].astype(np.int32)
                          x1, y1, w, h = coords[0], coords[1], coords[2], coords[3]
                          cv.rectangle(output_image, (x1, y1), (x1+w, y1+h), (0,0,255), 1)
                     except: pass
        elif isinstance(faces, tuple) and len(faces) > 1 and faces[1] is None:
             st.info("Không phát hiện thấy khuôn mặt nào trong ảnh.")
        else:
             st.warning(f"Định dạng kết quả detect không như mong đợi: {type(faces)}")


    except Exception as e_detect_img:
        st.error(f"Lỗi trong quá trình detect khuôn mặt trên ảnh: {e_detect_img}")
        import traceback
        st.code(traceback.format_exc())

    st.info("Kết thúc recognize_faces_in_image.")
    return output_image

# Chương trình chính
st.title("Nhận dạng khuôn mặt 5 người 🙂")
svc_model = load_svm_model(SVM_MODEL_PATH)
detector_model = load_face_detector(DETECTOR_MODEL_PATH)
recognizer_model = load_face_recognizer(RECOGNIZER_MODEL_PATH)

tab1, tab2, tab3 = st.tabs(["WebCam", "Video File", "Image"])

# Webcam
with tab1:
    st.header("Sử dụng Webcam 📹")
    run_realtime = st.button("Start Real-Time Detection", key="start_realtime")
    if run_realtime:
        # Đặt lại trạng thái dừng trước khi bắt đầu
        st.session_state['stop_processing_0'] = False
        st.info("Bắt đầu xử lý Real-Time...")
        process_video_stream(0)

# Video File
with tab2:
    st.header("Xử lý từ File Video 🎬")
    default_video_path = Path("./images/video_predict.mp4")

    if default_video_path.is_file():
        st.write("Xem trước video mặc định:")
        try:
            video_file = open(default_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            run_default = st.button("Start Processing Default Video", key="start_default_video")
            if run_default:
                st.session_state[f'stop_processing_{str(default_video_path)}'] = False
                st.info("Bắt đầu xử lý video mặc định...")
                process_video_stream(str(default_video_path))
        except Exception as e:
            st.error(f"Error reading or displaying default video: {e}")
    else:
        st.warning(f"Default video not found at: {default_video_path}")

    st.divider()
    uploaded_file = st.file_uploader("Hoặc tải lên file video khác", type=["mp4", "avi", "mov", "mkv"], key="file_uploader")
    if uploaded_file is not None:
        temp_video_path = Path("./temp_video_uploaded") / uploaded_file.name
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Xem trước video đã tải lên:")
        st.video(uploaded_file)
        run_uploaded = st.button("Start Processing Uploaded Video", key="start_uploaded_video")
        if run_uploaded:
            st.session_state[f'stop_processing_{str(temp_video_path)}'] = False
            st.info("Bắt đầu xử lý video tải lên...")
            process_video_stream(str(temp_video_path))

# Image
with tab3:
    st.header("Xử lý từ Ảnh Tĩnh 📷")
    uploaded_image_file = st.file_uploader(
        "Tải lên một hình ảnh",
        type=["jpg", "jpeg", "png", "bmp"],
        key="face_image_uploader"
    )

    if uploaded_image_file is not None:
        if not all([svc_model, detector_model, recognizer_model]):
             st.error("Một hoặc nhiều model cần thiết chưa được tải thành công. Không thể xử lý ảnh.")
        else:
            col1_img, col2_img = st.columns(2)
            try:
                # Đọc ảnh sử dụng PIL
                image_pil = Image.open(uploaded_image_file)
                with col1_img:
                     st.subheader("Ảnh gốc")
                     st.image(image_pil, caption=f"Ảnh tải lên: {uploaded_image_file.name}", use_container_width=True)

                # Convert PIL Image to OpenCV BGR format
                image_bgr = cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR)

                with st.spinner("Đang nhận dạng khuôn mặt trên ảnh..."):
                     image_result_bgr = recognize_faces_in_image(
                         image_bgr,
                         detector=detector_model,
                         recognizer=recognizer_model,
                         svc=svc_model
                     )

                with col2_img:
                     st.subheader("Kết quả nhận dạng")
                     image_result_rgb = cv.cvtColor(image_result_bgr, cv.COLOR_BGR2RGB)
                     st.image(image_result_rgb, caption="Ảnh đã xử lý", use_container_width=True)

            except Exception as e_img:
                st.error(f"Đã xảy ra lỗi khi đọc hoặc xử lý ảnh: {e_img}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Vui lòng tải lên một hình ảnh trong tab này để bắt đầu nhận dạng.")

