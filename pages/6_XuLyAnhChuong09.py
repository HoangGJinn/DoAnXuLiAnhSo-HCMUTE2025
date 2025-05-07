import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
try:
    from library.Chapter09 import Erosion, Dilation, Boundary, HoleFill, ConnectedComponent, CountRice
except ImportError:
    st.error("Lỗi: Không thể import hàm từ library/Chapter09.py. "
             "Hãy đảm bảo file này tồn tại trong thư mục library "
             "và đường dẫn sys.path (nếu cần) được thiết lập đúng.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi không mong muốn khi import: {e}")
    st.stop()

DEFAULT_IMAGE_DIR = Path("./images/ImageProcessingChapter09")


OPERATIONS_C9 = {
    "1. Erosion (Co - Kernel 45x45)": {"func": Erosion, "default_img": "Erosion_default.jpg"}, # Kernel cố định trong hàm gốc
    "2. Dilation (Giãn - Kernel 3x3)": {"func": Dilation, "default_img": "delation_default.png"}, # Kernel cố định
    "3. Boundary Extraction (Trích biên)": {"func": Boundary, "default_img": "boundary_default.jpg"}, # Kernel cố định
    "4. Hole Filling (Lấp lỗ)": {"func": HoleFill, "default_img": "HoleFill_default.jpg"},
    "5. Connected Components (TP Liên thông)": {"func": ConnectedComponent, "default_img": "connectedComponent_default.png"},
    "6. Count Rice (Đếm gạo)": {"func": CountRice, "default_img": "countRice_default.png"}
}
OPERATION_NAMES = list(OPERATIONS_C9.keys())


st.title("Xử lý Hình thái Ảnh (Chương 9) 🔬")
selected_operation_name = st.selectbox(
    "Chọn phép xử lý hình thái:",
    options=OPERATION_NAMES,
    key="c9_operation_select_orig_v4" # Key mới
)

operation_details = OPERATIONS_C9[selected_operation_name]
processing_function = operation_details["func"]
default_image_filename = operation_details.get("default_img")

img_input_for_processing = None
uploaded_file = st.file_uploader(
    "Tải lên ảnh",
    type=["bmp", "png", "jpg", "jpeg"],
    key="c9_uploader_orig_v4"
)
use_default = False
if uploaded_file is None and default_image_filename:
    if DEFAULT_IMAGE_DIR.is_dir():
         default_image_path_check = DEFAULT_IMAGE_DIR / default_image_filename
         if default_image_path_check.is_file():
              use_default = st.checkbox(f"Sử dụng ảnh mặc định ({default_image_filename})", value=True, key="c9_use_default_orig_v4")
         else: st.warning(f"Ảnh mặc định không tồn tại: {default_image_path_check}")
    else: st.warning(f"Thư mục ảnh mặc định không tìm thấy: {DEFAULT_IMAGE_DIR}")

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
    else: st.error(f"Lỗi: Không thể sử dụng ảnh mặc định: {default_image_path}")
else: st.warning("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mặc định.")


col1, col2 = st.columns(2)
img_original_pil = None

if input_image_path_or_buffer:
    try:
        img_original_pil = Image.open(input_image_path_or_buffer)
        img_bgr = cv2.cvtColor(np.array(img_original_pil), cv2.COLOR_RGB2BGR)

        img_input_for_processing = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(img_original_pil, caption=input_caption, use_container_width=True)
        with col2:
            st.subheader(f"Kết quả:")

    except Exception as e:
        st.error(f"Không thể đọc hoặc xử lý ảnh đầu vào: {e}")
        img_input_for_processing = None
else:
    img_input_for_processing = None

if img_input_for_processing is not None:
    st.divider()
    output_result = None

    try:
        with st.spinner("Đang xử lý ..."):
            output_result = processing_function(img_input_for_processing)

            with col2:
                if output_result is not None:
                    if isinstance(output_result, tuple) and len(output_result) == 2:
                        result_text, result_image = output_result
                        if isinstance(result_image, np.ndarray):
                            st.image(result_image, caption=f"Kết quả ảnh: {selected_operation_name}", use_container_width=True)
                        else: st.warning("Kết quả không chứa hình ảnh hợp lệ.")
                        if isinstance(result_text, str):
                             st.text_area("Kết quả đếm:", result_text, height=70)

                    elif isinstance(output_result, np.ndarray):
                         if output_result.ndim == 3 and output_result.shape[2] == 3: # Ảnh màu (từ HoleFill)
                              img_rgb = cv2.cvtColor(output_result, cv2.COLOR_BGR2RGB)
                              st.image(img_rgb, caption=f"Kết quả: {selected_operation_name}", use_container_width=True)
                         elif output_result.ndim == 2:
                              st.image(output_result, caption=f"Kết quả: {selected_operation_name}", use_container_width=True)
                         else: st.error("Định dạng ảnh kết quả không hợp lệ.")
                    else:
                        st.error("Hàm xử lý không trả về kết quả hợp lệ (ảnh hoặc tuple).")
                else:
                    st.warning("Hàm xử lý không trả về kết quả.")

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý '{selected_operation_name}': {e}")
        import traceback
        st.code(traceback.format_exc())
else:
     with col1:
          st.info("Chưa có ảnh đầu vào để xử lý.")
