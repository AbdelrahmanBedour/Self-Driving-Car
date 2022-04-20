import numpy as np
import cv2
#import matplotlib.pyplot as plt
#from moviepy.editor import VideoFileClip

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def Edge_Detection(img):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hls[:,:,2]
    v_channel=  hsv[:,:,2]
    _,v_channel=cv2.threshold(v_channel,180, 255, cv2.THRESH_BINARY_INV)
    canny_s =cv2.Canny(remove_noise(s_channel,5),50,150)
    canny_v =cv2.Canny(remove_noise(v_channel,7),50,150)
    canny_output = canny_s | canny_v
    return canny_output


def perspective_transform(image):
    height = image.shape[0]
    width = image.shape[1]
    # Quadrangle vertices coordinates in the source image
    s1 = [width // 2 - int(width*0.07) , height * 0.65]
    s2 = [width // 2 + int(width*0.08), height * 0.65]
    s3 = [int(0.1*width), height]
    s4 = [width , height]
    src = np.float32([s1, s2, s3, s4])
    # Quadrangle vertices coordinates in the destination image
    d1 = [0, 0]
    d2 = [width, 0]
    d3 = [0, height]
    d4 = [width, height]
    dst = np.float32([d1, d2, d3, d4])
    # Given src and dst points we calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def inv_perspective_transform(image):
    height = image.shape[0]
    width = image.shape[1]
    # Quadrangle verties coordinates in the source image
    d1 = [width // 2 - int(width*0.07) , height * 0.65] ## 0.65 --> 0.55
    d2 = [width // 2 + int(width*0.08), height * 0.65]## 0.65
    d3 = [int(0.1*width), height]
    d4 = [width , height]
    dst = np.float32([d1, d2, d3, d4])
    # Quadrangle verties coordinates in the destination image
    s1 = [0, 0]
    s2 = [width, 0]
    s3 = [0, height]
    s4 = [width, height]
    src = np.float32([s1, s2, s3, s4])
    # Given src and dst points we calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image
    unwrap_m = cv2.warpPerspective(image, M, (width, height))
    # We also calculate the oposite transform
    return unwrap_m

