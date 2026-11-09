import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="이미지 변환기", layout="wide")
st.title("이미지 변환기")
file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])

if file:
    img = np.array(Image.open(file).convert("RGB"))
    h, w = img.shape[:2]
    menu = ["Original", "Grayscale", "Crop", "Rotate 30°", "Gaussian Blur", "Median Blur", "Homography"]
    mode = st.sidebar.selectbox("변환 방식", menu)
    result = img

    if mode == "Grayscale":
        result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif mode == "Crop":
        x1, x2 = st.sidebar.slider("X 범위", 0, w, (0, w))
        y1, y2 = st.sidebar.slider("Y 범위", 0, h, (0, h))
        if x1 < x2 and y1 < y2:
            result = img[y1:y2, x1:x2]
        else:
            st.warning("올바른 자르기 범위를 선택하세요.")
    elif mode == "Rotate 30°":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1)
        nw = int(h * abs(m[0, 1]) + w * abs(m[0, 0]))
        nh = int(h * abs(m[0, 0]) + w * abs(m[0, 1]))
        m[0, 2] += (nw - w) / 2
        m[1, 2] += (nh - h) / 2
        result = cv2.warpAffine(img, m, (nw, nh))
    elif mode == "Gaussian Blur":
        result = cv2.GaussianBlur(img, (5, 5), 0)
    elif mode == "Median Blur":
        result = cv2.medianBlur(img, 5)
    elif mode == "Homography":
        corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        pts = [[st.sidebar.number_input(f"{name} X", 0, w - 1, x),
                st.sidebar.number_input(f"{name} Y", 0, h - 1, y)]
               for name, (x, y) in zip(["좌상단", "우상단", "우하단", "좌하단"], corners)]
        src = np.float32(pts)
        tw = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
        th = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
        if tw > 1 and th > 1 and abs(cv2.contourArea(src)) > 1:
            dst = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
            result = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (tw, th))
        else:
            st.warning("원근 변환 좌표를 확인하세요.")

    left, right = st.columns(2)
    left.image(img, caption="원본 이미지", use_container_width=True)
    right.image(result, caption="변환 이미지", use_container_width=True)
    encoded = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                           if result.ndim == 3 else result)[1]
    st.download_button("이미지 다운로드", encoded.tobytes(), "result.png", "image/png")
else:
    st.info("이미지를 업로드하세요.")