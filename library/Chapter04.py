import numpy as np
import cv2 as cv

L = 256

# Tạo một checkerboard để nhân với số (-1)^(x+y)
def create_checkerboard(rows, cols):
    checkerboard = np.ones((rows, cols), np.float32)
    checkerboard[::2, 1::2] = -1
    checkerboard[1::2, ::2] = -1
    return checkerboard

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)

    # B1: Tạo ảnh mới có kích thước PxQ và thêm 0 ở phần mở rộng
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin
    fp /= (L - 1)

    # B2: Nhân checkerboard để dời vào tâm ảnh
    checkerboard_orig = create_checkerboard(M, N)
    fp[:M, :N] *= checkerboard_orig

    # B3: DFT
    F = cv.dft(fp, flags=cv.DFT_COMPLEX_OUTPUT)
    # B4: Tính Spectrum
    S = cv.magnitude(F[:, :, 0], F[:, :, 1])
    S = np.clip(S, 0, L - 1)
    S = S.astype(np.uint8)
    return S


def FrequencyFilter(imgin):
    M, N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)

    # B1: Tạo ảnh mới có kích thước PxQ và thêm số 0 ở phần mở rộng
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin
    # B2: Nhân checkerboard để dời vào tâm ảnh
    checkerboard_orig = create_checkerboard(M, N)
    fp[:M, :N] *= checkerboard_orig
    # B3: DFT
    F = cv.dft(fp, flags=cv.DFT_COMPLEX_OUTPUT)
    # B4: Tạo bộ lọc High Pass Butterworth
    H = np.zeros((P, Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u - P // 2) ** 2 + (v - Q // 2) ** 2)
            if Duv > 0:
                H[u, v] = 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
    # B5: nhân từng đôi G
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u, v, 0] = F[u, v, 0] * H[u, v]
            G[u, v, 1] = F[u, v, 1] * H[u, v]
    # B6: IDFT
    g = cv.idft(G, flags=cv.DFT_SCALE)
    gp = g[:, :, 0] # phần thực
    for x in range(0, P):
        for y in range(0, Q):
            if (x + y) % 2 == 1:
                gp[x, y] = -gp[x, y]

    imgout = gp[0:M, 0:N]
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout


def CreateNotchRejectFilter():
    P = 250
    Q = 180
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P, Q), np.float32)
    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            # u1 v1
            Temp = np.sqrt((u - u1) ** 2 + (v - v1) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0
            Temp = np.sqrt((u - (P - u1)) ** 2 + (v - (Q - v1)) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0

            # u2, v2
            Temp = np.sqrt((u - u2) ** 2 + (v - v2) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0
            Temp = np.sqrt((u - (P - u2)) ** 2 + (v - (Q - v2)) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0

            # u3, v3
            Temp = np.sqrt((u - u3) ** 2 + (v - v3) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0
            Temp = np.sqrt((u - (P - u3)) ** 2 + (v - (Q - v3)) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0

            # u4, v4
            Temp = np.sqrt((u - u4) ** 2 + (v - v4) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0
            Temp = np.sqrt((u - (P - u4)) ** 2 + (v - (Q - v4)) ** 2)
            if Temp > 0:
                h = h * 1.0 / (1.0 + np.power(D0 / Temp, 2 * n))
            else:
                h = h * 0.0
            H[u, v] = h
    return H


def DrawNotchRejectFilter():
    H = CreateNotchRejectFilter()
    H = H * (L - 1)
    H = H.astype(np.uint8)
    return H


def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    # B1:
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin
    # B2:
    checkerboard_orig = create_checkerboard(M, N)
    fp[:M, :N] *= checkerboard_orig
    # B3:
    F = cv.dft(fp, flags=cv.DFT_COMPLEX_OUTPUT)
    # B4:
    H = CreateNotchRejectFilter()
    # B5:
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u, v, 0] = F[u, v, 0] * H[u, v]
            G[u, v, 1] = F[u, v, 1] * H[u, v]
    # B6:
    g = cv.idft(G, flags=cv.DFT_SCALE)
    gp = g[:, :, 0]
    for x in range(0, P):
        for y in range(0, Q):
            if (x + y) % 2 == 1:
                gp[x, y] = -gp[x, y]

    imgout = gp[0:M, 0:N]
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout
