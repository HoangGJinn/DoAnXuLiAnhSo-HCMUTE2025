import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


try:
    from library.Chapter05 import CreateMotion, Demotion, DemotionWeiner
except ImportError:
    st.error("Lỗi: Không thể import hàm từ library/chapter5.py. "
             "Hãy đảm bảo file chapter5.py gốc của bạn nằm trong thư mục library "
             "và đường dẫn sys.path (nếu cần) được thiết lập đúng.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi không mong muốn khi import: {e}")
    st.stop()


DEFAULT_IMAGE_DIR = Path("./images/ImageProcessingChapter05")


OPERATIONS_C5 = {
    "1. Tạo Nhiễu Ảnh (Motion Blur)": {"func_name": "CreateMotion", "default_img": "MotionBlur_default.png"},
    "2. Lọc Ảnh Ít Nhiễu (Inverse Filter - Demotion)": {"func_name": "Demotion", "default_img": "InverseFilter_default.png"},
    "3. Lọc Ảnh Nhiều Nhiễu (Median 7x7 + Demotion)": {"func_name": "Demotion_Median", "default_img": "MedianDemolition_default.png"},
    "4. Lọc Ảnh Nhiễu (Wiener - Median 7x7 + DemotionWeiner)": {"func_name": "DemotionWeiner_Median", "default_img": "MedianDemolition_default.png"}
}
OPERATION_NAMES = list(OPERATIONS_C5.keys())


st.title("Khôi phục ảnh (Chương 5) 🔧")
selected_operation_name = st.selectbox(
    "Chọn chức năng:",
    options=OPERATION_NAMES,
    index=0,
    key="c5_operation_select_orig_v3"
)

operation_details = OPERATIONS_C5[selected_operation_name]
processing_function_name = operation_details["func_name"]
default_image_filename = operation_details.get("default_img")

img_input_gray = None
uploaded_file = st.file_uploader(
    "Tải lên ảnh",
    type=["bmp", "png", "jpg", "jpeg"],
    key="c5_uploader_orig_v3"
)
use_default = False
if uploaded_file is None and default_image_filename:
    if DEFAULT_IMAGE_DIR.is_dir():
         default_image_path_check = DEFAULT_IMAGE_DIR / default_image_filename
         if default_image_path_check.is_file():
              use_default = st.checkbox(f"Sử dụng ảnh mặc định ({default_image_filename})", value=True, key="c5_use_default_orig_v3")
         else:
              st.warning(f"Ảnh mặc định không tồn tại: {default_image_path_check}")
    else:
         st.warning(f"Thư mục ảnh mặc định không tìm thấy: {DEFAULT_IMAGE_DIR}")

input_image_path_or_buffer = None
input_caption = "Chưa chọn ảnh"
if uploaded_file:
    input_image_path_or_buffer = uploaded_file
    input_caption = f"Ảnh tải lên: {uploaded_file.name}"
elif use_default and default_image_filename:
    default_image_path = DEFAULT_IMAGE_DIR / default_image_filename
    if default_image_path.is_file():
        input_image_path_or_buffer = default_image_path
        input_caption = f"Ảnh mặc định: {default_image_filename}"
    else:
        st.error(f"Lỗi: Không thể sử dụng ảnh mặc định vì file không tồn tại: {default_image_path}")
else:
     st.warning("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mặc định.")

col1, col2 = st.columns(2)

if input_image_path_or_buffer:
    try:
        img_pil = Image.open(input_image_path_or_buffer)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_input_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        with col1:
            st.subheader("Ảnh gốc:")
            st.image(img_pil, caption=input_caption, use_container_width=True)

        with col2:
            st.subheader("Kết quả:")

    except Exception as e:
        st.error(f"Không thể đọc hoặc xử lý ảnh đầu vào: {e}")
        img_input_gray = None
else:
    img_input_gray = None

if img_input_gray is not None:
    st.divider()

    output_image = None
    try:
        with st.spinner("Đang xử lý... "):
            target_function = None
            input_for_func = img_input_gray # Mặc định dùng ảnh gốc đã chuyển xám

            if processing_function_name == "Demotion_Median":
                 st.info("Áp dụng Median Blur (kernel 7x7) trước khi gọi Demotion...")
                 input_for_func = cv2.medianBlur(img_input_gray, 7) # Ảnh đầu vào là ảnh đã blur
                 if "Demotion" in globals():
                     target_function = globals()["Demotion"]
                 else:
                      st.error("Lỗi: Hàm 'Demotion' không được import.")
            elif processing_function_name == "DemotionWeiner_Median":
                 st.info("Áp dụng Median Blur (kernel 7x7) trước khi gọi DemotionWeiner...")
                 input_for_func = cv2.medianBlur(img_input_gray, 7) # Ảnh đầu vào là ảnh đã blur
                 if "DemotionWeiner" in globals():
                     target_function = globals()["DemotionWeiner"]
                 else:
                      st.error("Lỗi: Hàm 'DemotionWeiner' không được import.")
            else:
                 # Gọi các hàm khác trực tiếp
                 if processing_function_name in globals():
                     target_function = globals()[processing_function_name]
                 else:
                     st.error(f"Lỗi: Hàm '{processing_function_name}' không được import.")

            # Thực thi hàm nếu đã tìm thấy
            if target_function:
                 output_image = target_function(input_for_func)
            else:
                 st.error("Không thể thực thi do lỗi import hàm.")

            with col2:
                if output_image is not None and isinstance(output_image, np.ndarray):
                    st.image(output_image, caption=f"Kết quả: {selected_operation_name}", use_container_width=True)
                elif target_function:
                    st.warning("Hàm xử lý đã chạy nhưng không trả về kết quả hợp lệ.")

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý '{selected_operation_name}': {e}")
        import traceback
        st.code(traceback.format_exc())
else:
     with col1:
          st.info("Chưa có ảnh đầu vào để xử lý.")
