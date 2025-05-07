import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from library.Chapter04 import Spectrum, FrequencyFilter, DrawNotchRejectFilter, RemoveMoire
except ImportError:
    st.error("Lỗi: Không thể import các hàm từ library/Chapter04.py.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi xảy ra khi import: {e}")
    st.stop()

DEFAULT_IMAGE_DIR_C4 = Path("./images/ImageProcessingChapter04")

OPERATIONS_C4 = {
    "1. Phổ tần số (Spectrum)": {"func": Spectrum, "default_img": "Spectrum_default.png", "needs_input": True},
    "2. Lọc tần số (High Pass Butterworth)": {"func": FrequencyFilter, "default_img": "HighpassBufferworth_default.png", "needs_input": True},
    "3. Vẽ bộ lọc chắn dải (Notch Reject Filter)": {"func": DrawNotchRejectFilter, "default_img": None, "needs_input": False}, # Không cần ảnh đầu vào
    "4. Loại bỏ nhiễu Moire (dùng Notch Filter)": {"func": RemoveMoire, "default_img": "Moire_default.png", "needs_input": True},
}
OPERATION_NAMES_C4 = list(OPERATIONS_C4.keys())


st.title("Xử lý ảnh trong miền tần số (Chapter 4) 📊")

selected_operation_name_c4 = st.selectbox(
    "Chọn phép toán trong miền tần số:",
    options=OPERATION_NAMES_C4,
    key="c4_op_select_exact_c9_v1" # Key duy nhất
)

operation_details_c4 = OPERATIONS_C4[selected_operation_name_c4]
processing_function_c4 = operation_details_c4["func"]
default_image_filename_c4 = operation_details_c4.get("default_img")
needs_input_image = operation_details_c4.get("needs_input", True)

img_input_for_processing_c4 = None
img_original_pil_c4 = None
input_image_path_or_buffer_c4 = None
input_caption_c4 = "Chưa chọn ảnh"
uploaded_file_c4 = None
use_default_c4 = False
columns_defined = False # Cờ để biết st.columns đã được gọi chưa
can_process_without_input = False # Cờ cho phép xử lý khi không cần input

if needs_input_image:
    uploaded_file_c4 = st.file_uploader(
        "Tải ảnh lên",
        type=["bmp", "png", "jpg", "jpeg", "tif"],
        key="c4_uploader_exact_c9_v1"
    )
    if uploaded_file_c4 is None and default_image_filename_c4:
        if DEFAULT_IMAGE_DIR_C4.is_dir():
             default_image_path_check_c4 = DEFAULT_IMAGE_DIR_C4 / default_image_filename_c4
             if default_image_path_check_c4.is_file():
                  use_default_c4 = st.checkbox(f"Sử dụng ảnh mặc định ({default_image_filename_c4})", value=True, key="c4_use_default_exact_c9_v1")

    if uploaded_file_c4:
        input_image_path_or_buffer_c4 = uploaded_file_c4
        input_caption_c4 = f"Đã tải lên: {uploaded_file_c4.name}"
    elif use_default_c4 and default_image_filename_c4:
        default_image_path_c4 = DEFAULT_IMAGE_DIR_C4 / default_image_filename_c4
        if default_image_path_c4.is_file():
            input_image_path_or_buffer_c4 = str(default_image_path_c4)
            input_caption_c4 = f"Mặc định: {default_image_filename_c4}"
        else:
             st.error(f"Lỗi: Không thể sử dụng ảnh mặc định: {default_image_path_c4}")
             input_image_path_or_buffer_c4 = None

else:
    can_process_without_input = True

col1_c4 = None
col2_c4 = None

if needs_input_image:
    if input_image_path_or_buffer_c4:
        try:
            img_original_pil_c4 = Image.open(input_image_path_or_buffer_c4)
            img_bgr_c4 = cv2.cvtColor(np.array(img_original_pil_c4), cv2.COLOR_RGB2BGR)
            img_input_for_processing_c4 = cv2.cvtColor(img_bgr_c4, cv2.COLOR_BGR2GRAY) # Luôn chuyển sang xám

            col1_c4, col2_c4 = st.columns(2)
            columns_defined = True
            with col1_c4:
                st.subheader("Ảnh gốc")
                st.image(img_original_pil_c4, caption=input_caption_c4, use_container_width=True)
            with col2_c4:
                st.subheader(f"Kết quả:")

        except Exception as e:
            st.error(f"Không thể đọc hoặc xử lý ảnh đầu vào: {e}")
            img_input_for_processing_c4 = None

else:

     col1_c4, col2_c4 = st.columns(2)
     columns_defined = True
     with col1_c4:
          st.subheader("Đầu vào")
          st.info("Phép toán này không yêu cầu ảnh đầu vào.")
     with col2_c4:
          st.subheader(f"Kết quả:")

if img_input_for_processing_c4 is not None or can_process_without_input:
    output_result_c4 = None
    try:
        with st.spinner("Đang xử lý..."):
            if img_input_for_processing_c4 is not None:
                output_result_c4 = processing_function_c4(img_input_for_processing_c4)
            elif can_process_without_input:
                output_result_c4 = processing_function_c4()
            else:
                 st.warning("Trạng thái không xác định, không thể xử lý.")

        if columns_defined and col2_c4:
             with col2_c4:
                if output_result_c4 is not None:
                    if isinstance(output_result_c4, np.ndarray):
                        if output_result_c4.ndim == 2:
                            st.image(output_result_c4, caption=f"Kết quả: {selected_operation_name_c4}", use_container_width=True)
                        else:
                             st.error("Hàm xử lý trả về định dạng ảnh không mong muốn.")
                    else:
                        st.error("Hàm xử lý không trả về kết quả dạng ảnh (numpy array).")
                else:
                    st.warning("Hàm xử lý không trả về kết quả.")

        else:
             st.error("Lỗi cấu trúc: Không thể hiển thị kết quả vì cột chưa được định nghĩa.")

    except Exception as e:
        error_container = col2_c4 if columns_defined and col2_c4 else st
        error_container.error(f"Lỗi trong quá trình xử lý '{selected_operation_name_c4}': {e}")
        import traceback
        error_container.code(traceback.format_exc())

else:
    if columns_defined and col1_c4:
         with col1_c4:
              st.info("Chưa có ảnh đầu vào hợp lệ để xử lý.")
    elif needs_input_image:
         st.warning("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mặc định.")