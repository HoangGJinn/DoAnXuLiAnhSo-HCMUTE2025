import cv2
import numpy as np
L = 256

def _ensure_binary_or_gray(imgin, threshold_val=128):
    #Đảm bảo ảnh có màu xám và sẵn tiện tuỳ chỉnh ngưỡng
    if imgin.ndim == 3 and imgin.shape[2] == 3:
        img_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    elif imgin.ndim == 2:
        img_gray = imgin
    else:
        print(f"Lỗi {imgin.ndim}. Đợi chuyển sang màu xám")
        try:
            if imgin.shape[2] == 4:
                 img_gray = cv2.cvtColor(imgin, cv2.COLOR_BGRA2GRAY)
            else:
                 raise ValueError("Hình ảnh không được hỗ trợ")
        except Exception as e:
             print(f"Không thể chuyển sang xám {e}")
             return None

    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binary

def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    imgout = cv2.erode(imgin,w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgout = cv2.dilate(imgin, w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp
    return imgout

def HoleFill(imgin):
    img_binary = _ensure_binary_or_gray(imgin)
    if img_binary is None: return None
    img_floodfill = img_binary.copy()

    # Dùng mask cho thuật toán flooding
    # Size cần lớn hơn file ảnh 2 pixel.
    h, w = img_binary.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill từ (0, 0) đến rìa ảnh -- giả sử có viền dài 255
    cv2.floodFill(img_floodfill, mask, (0, 0), 255);
    # Dùng mask và thuật toán flooding
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    # Ghép 2 ảnh
    imgout = img_binary | img_floodfill_inv
    return imgout

def ConnectedComponent(imgin):
    # Đảm bảo ảnh xám và tuỳ chỉnh ngưỡng
    if imgin.ndim == 3 and imgin.shape[2] == 3:
        img_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    elif imgin.ndim == 2:
        img_gray = imgin
    else:
        print("Lỗi: Yêu cầu ảnh xám hoặc ảnh màu BGR.")
        return np.zeros_like(imgin) if imgin is not None else None

    nguong = 200
    _, temp = cv2.threshold(img_gray, nguong, L-1, cv2.THRESH_BINARY)
    # Tìm thành phần liên thông
    n, label = cv2.connectedComponents(temp)
    # Tính diện tích (giữ nguyên logic đếm pixel)
    a = np.zeros(n, np.int32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0: # Chỉ đếm các thành phần khác nền
                a[r] = a[r] + 1

    img_with_text = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    num_components = n - 1
    s = 'Co %d thanh phan lien thong' % (num_components)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    start_y = 25
    line_spacing = 20
    cv2.putText(img_with_text, s, (10, start_y), font_face, font_scale, text_color, thickness)

    if num_components > 0:
        for i in range(1, n):
            s = '%2d: %d px' % (i, a[i])
            current_y = start_y + i * line_spacing
            cv2.putText(img_with_text, s, (10, current_y), font_face, font_scale, text_color, thickness)
    return img_with_text


def CountRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d hat gao' % (dem-1)
    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color
    max = a[1]
    for r in range(2, dem):
        if a[r] > max:
            max = a[r]
    xoa = np.array([], np.int32)
    for r in range(1, dem):
        if a[r] < 0.5*max:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                r = r - color
                if r in xoa:
                    label[x,y] = 0
    return text,temp


