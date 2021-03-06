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





def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

# ------------------------- Draw Lanes----------------------------------------#
# ---------------------------------------------------------------------------#

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (34, 139, 34))
    inv_perspective = inv_perspective_transform(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

# ---------------------------------------------------------------------------#


# ------------------------- Get Curve----------------------------------------#
# ---------------------------------------------------------------------------#
def get_curve(img, leftx, rightx):
    # Generate y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    # calculate meters per pixel (field of view in meters / image width)
    ym_per_pix = 30.5 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)  ## Scaling for real world
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # calculate car position
    car_pos = img.shape[1] / 2

    # calculate lane center
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2

    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)

# ---------------------------------------------------------------------------#



# ----------------------- Pipeline in Debugging Mode-------------------------#
# ---------------------------------------------------------------------------#
def pipeline_deb_mode(img):
    img_copy = img
    filters = []
    edge_output = Edge_Detection(img)  ## Output of Canny Edge Detection + Filtering
    warped = perspective_transform(edge_output)  ## Output of Perspective Transform
    sliding_out, curves, _, _ = sliding_window(warped, draw_windows=False)  ## Output of Sliding Window

    curve_radius = get_curve(img, curves[0], curves[1])  ## Return The radius of right and left curves

    lane_curve = np.mean(
        [curve_radius[0], curve_radius[1]])  ## Calculating the mean value of the right and left curves radius
    img = draw_lanes(img, curves[0], curves[1])

    ####

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontColor = (255, 255, 255)
    fontSize = 1.3
    cv2.putText(img, 'Radius of Curvature: {:.0f} m'.format(lane_curve), (30, 100), font, fontSize, fontColor, 2)
    cv2.putText(img, 'Vehicle is {:.4f} m of the center'.format(curve_radius[2]), (30, 150), font, fontSize, fontColor,
                2)

    ###
    src = np.float32([(0.43, 0.65), (0.1, 1), (1, 1), (0.58, 0.65)])
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    src = src.reshape((-1, 1, 2)).astype(np.int32)

    cv2.polylines(img_copy, [src], True, color=(255, 0, 0), thickness=4)
    ###

    filters.append(sliding_out)
    filters.append(warped)
    filters.append(img_copy)
    filters.append(edge_output)

    return show_debugging(img, filters)

# ---------------------------------------------------------------------------#


# ----------------------- Pipeline in Normal Mode ----------------------------#
# ----------------------------------------------------------------------------#

def pipeline_normal_mode(img):
    edge_output = Edge_Detection(img)  ## Output of Canny Edge Detection + Filtering
    warped = perspective_transform(edge_output)  ## Output of Perspective Transform
    _, curves, _, _ = sliding_window(warped, draw_windows=False)  ## Output of Sliding Window
    curve_radius = get_curve(img, curves[0], curves[1])  ## Return The radius of right and left curves
    lane_curve = np.mean(
        [curve_radius[0], curve_radius[1]])  ## Calculating the mean value of the right and left curves radius
    img = draw_lanes(img, curves[0], curves[1])  ## Draw the Lane

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontColor = (255, 255, 255)
    fontSize = 1.3
    cv2.putText(img, 'Radius of Curvature: {:.0f} m'.format(lane_curve), (30, 100), font, fontSize, fontColor, 2)
    cv2.putText(img, 'Vehicle is {:.4f} m of the center'.format(curve_radius[2]), (30, 150), font, fontSize, fontColor,
                2)

    return img

# ---------------------------------------------------------------------------#


# ------------ Function That Show Stacked Windows ----------------------------#
# ----------------------------------------------------------------------------#

def show_debugging(img, filters):
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontColor = (255, 255, 255)
    fontSize = 0.7
    # cv2.putText(img, 'Radius of Curvature: {:.0f} m'.format(lane_curve), (30, 100), font, fontSize, fontColor, 2)
    # cv2.putText(img, 'Vehicle is {:.4f} m of the center'.format(curve_radius[2]), (30, 150), font, fontSize, fontColor, 2)

    for i, filt in enumerate(filters):
        if (len(filt.shape) < 3):
            filters[i] = cv2.cvtColor(filt, cv2.COLOR_GRAY2RGB)

    if len(filters) == 4:
        img_out = np.zeros((960, 1624, 3), dtype=np.uint8)
        img_out[0:576, 0:1024, 0:3] = cv2.resize(img, (1024, 576))  ## Real Image
        # ----------------------------------------
        img_out[576:, 0:512, 0:3] = cv2.resize(filters[0], (512, 384))
        img_out[576:, 512:1024, 0:3] = cv2.resize(filters[1], (512, 384))
        img_out[0:480, 1024:, 0:3] = cv2.resize(filters[2], (600, 480))
        img_out[480:, 1024:, 0:3] = cv2.resize(filters[3], (600, 480))

        cv2.putText(img_out, 'Sliding Window Result', (10, 585), font, fontSize, fontColor, 2)
        cv2.putText(img_out, 'Bird Eye View ', (530, 585), font, fontSize, fontColor, 2)
        cv2.putText(img_out, 'Edge Detection', (1035, 490), font, fontSize, fontColor, 2)

    if len(filters) == 5:
        img_out = np.zeros((864, 1536, 3), dtype=np.uint8)
        img_out[0:576, 0:1024, 0:3] = cv2.resize(img, (1024, 576))
        # ----------------------------------------
        img_out[576:, 0:512, 0:3] = cv2.resize(filters[0], (512, 288))
        img_out[576:, 512:1024, 0:3] = cv2.resize(filters[1], (512, 288))
        img_out[576:, 1024:, 0:3] = cv2.resize(filters[2], (512, 288))
        img_out[:288, 1024:, 0:3] = cv2.resize(filters[3], (512, 288))
        img_out[288:576, 1024:, 0:3] = cv2.resize(filters[4], (512, 288))

    return img_out

# ---------------------------------------------------------------------------#
