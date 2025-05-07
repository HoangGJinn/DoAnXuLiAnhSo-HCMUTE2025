
import cv2
import numpy as np
import matplotlib.pyplot as plt

L = 256

def Negative(imgin):
    # M: height, N: width
    M, N = imgin.shape
    # Tạo ra ảnh imgout có kích thuước bằng imgin và có màu đen
    imgout = np.zeros((M, N), np.uint8)
    # Quét từng điểm ảnh
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = np.uint8(s)
    return imgout

def Logarit(imgin):
    if imgin.ndim > 2: # Đảm bảo là ảnh xám
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    img_float = imgin.astype(np.float32)
    c = (L - 1) / np.log(L)
    imgout_float = c * np.log1p(img_float)
    imgout = np.clip(imgout_float, 0, L - 1).astype(np.uint8)
    return imgout


def Power(imgin, gamma=5.0): # Gamma có thể là tham số
    # M: height, N: width
    M, N = imgin.shape
    # Tạo ra ảnh imgout có kích thuước bằng imgin và có màu đen
    imgout = np.zeros((M,N), np.uint8)
    c = np.power(L-1,1-gamma)
    # Quét từng điểm ảnh
    for x in range (0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c * np.power(r, gamma)
            imgout[x, y] = np.uint8(s)
    return imgout

# Biến đổi tuyến tính (đen ít thì đen nhiều, trắng ít thì trắng nhiều)
def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            # Đoạn I
            if r < r1:
                s = 1.0 * s1/r1 * r
            #Đoạn II
            elif r < r2:
                s = 1.0*(s2-s1)/(r2-r1) * (r - r1) + s1
            #Đoạn III
            else:
                s = 1.0*(L - 1 - s1)/(L - 1 - r1) * (r - r1) + s2
            imgout[x,y] = np.uint8(s)
    return imgout

def CalculateAndShowHistogram(imgin):
    if imgin.ndim > 2:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    # Tính histogram bằng OpenCV
    hist = cv2.calcHist([imgin], [0], None, [L], [0, L])
    # Tạo figure bằng Matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist, color='blue')
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Number of Pixels")
    ax.set_xlim([0, L-1])
    ax.grid(True)
    plt.tight_layout()
    return fig

def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = 1.0 * h / (M * N)
    s = np.zeros(L, np.float32)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    return imgout

def HistEqualColor(imgin):
    # Ảnh của opencv là ảnh BGR
    # Ảnh của pillow là ảnh RGB
    img_b = imgin[:,:,0]
    img_g = imgin[:, :, 1]
    img_r = imgin[:, :, 2]

    img_b = cv2.equalizeHist(img_b)
    img_g = cv2.equalizeHist(img_g)
    img_r = cv2.equalizeHist(img_r)

    imgout = imgin.copy()
    imgout[:,:,0] = img_b
    imgout[:, :, 1] = img_g
    imgout[:, :, 2] = img_r
    return imgout

def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 3
    for x in range (a, M-a):
        for y in range (b, N - b):
            w = imgin[x-a:x+a+1, y-b:y+b+1]
            w = cv2.equalizeHist(w)
            imgout[x,y] = w[a, b]
    return imgout

def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    mG, sigmaG = cv2.meanStdDev(imgin)
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    C = 22.8
    for x in range(a, M - a):
        for y in range (b, N - b):
            w = imgin[x-a:x+a+1, y-b:y+b+1]
            msxy, sigmasxy = cv2.meanStdDev(w)
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*imgin[x,y])
            else:
                imgout[x,y] = imgin[x,y]
    return imgout


def MySmoothBox(imgin, kernel_size=21):
    m = 21
    n = 21
    w = np.zeros((m,n), np.float32) + np.float32(1.0 / (m*n))
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1,w)
    return imgout

def Gauss(imgin, sigma=7.0):
    m = 7
    n = 43
    a = m//2
    b = n//2
    w = np.zeros((m,n), np.float32)
    for s in range (-a,a+1):
        for t in range (-b, b+1):
            w[s+a,t+b] = np.exp(-(s*s + t*t)/(sigma*sigma))
    K = 0
    for s in range (0,m):
        for t in range (0,n):
            K = K+ w[s,t]
    w = w/K
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1,w)
    return imgout

def Hubble(imgin, threshold=65): # Còn gọi là ThreShould (Phân ngưỡng)
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    temp = cv2.boxFilter(imgin, cv2.CV_8UC1, (15, 15))
    for x in range(0, M):
        for y in range(0, N):
            r = temp[x, y]
            if r > threshold:
                s = 255
            else:
                s = 0
            imgout[x, y] = np.uint8(s)
    return imgout

def MyMedianFilter(imgin, kernel_size=3):
    if imgin.ndim != 2:
         imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    # Đảm bảo kernel lẻ
    ksize = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    imgout = cv2.medianBlur(imgin, ksize)
    return imgout

def Sharp(imgin):
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]], np.float32)
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
     if imgin.ndim > 2:
         imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
     # Tính Sobel (đạo hàm cấp 1 theo x, y)
     grad_x = cv2.Sobel(imgin, cv2.CV_32F, 1, 0, ksize=3)
     grad_y = cv2.Sobel(imgin, cv2.CV_32F, 0, 1, ksize=3)
     # Tính độ lớn
     magnitude = cv2.magnitude(grad_x, grad_y)
     # Chuẩn hóa về 0-255
     imgout = cv2.normalize(magnitude, None, 0, L-1, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
     return imgout
