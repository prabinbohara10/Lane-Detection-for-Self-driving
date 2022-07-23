import sys
sys.modules[__name__].__dict__.clear()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    return gray

def gaussian_blur(img, kernel_size):
    gaussian_blur= cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) 
    # plt.imshow(gaussian_blur)
    # plt.show()
    return gaussian_blur

def canny(img, low_threshold, high_threshold):
    # plt.imshow(img)
    # plt.show()
    
    canny_image=cv2.Canny(img, low_threshold, high_threshold)
    # plt.imshow(canny_image)
    # plt.show()
    return canny_image



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def extrapolate(x,y):
    z = np.polyfit(x, y, 1)
    f = np.poly1d(z)
    #for i in range(min(x), max(x)):
    #    plt.plot(i, f(i), 'go')
    #plt.show()
    x_new = np.linspace(min(x), max(x), 10).astype(int)
    y_new = f(x_new).astype(int)
    points_new = list(zip(x_new, y_new))
    px, py = points_new[0]
    cx, cy = points_new[-1]
    return px, py, cx, cy

def draw_lines(img, lines, color=[255, 0, 0], thickness=2): 
    height, width, channels = img.shape
    # print("height haru")
    # print (height, width, channels)

    xp = []
    yp = []
    xn = []
    yn = []
    m = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), [0,255,0], thickness)
            m_old = m
            m = ((y2-y1)/(x2-x1))
            if  m > 0.5:
                xp += [x1, x2]
                yp += [y1, y2]
            elif m < -0.5:
                xn += [x1, x2]
                yn += [y1, y2]
    #if not a:
    #  print("List is empty")
    if len(xp)>0:
        pxp, pyp, cxp, cyp = extrapolate(xp,yp)  #right side (from top to bottom)
        m = ((cyp-pyp)/(cxp-pxp))
        if  abs(m) > 0.5:
            cv2.line(img, (pxp, pyp), (cxp, cyp), color, thickness)
    if len(xn)>0:
        pxn, pyn, cxn, cyn = extrapolate(xn,yn)  #left side (from top to bottom)
        m = ((cyn-pyn)/(cxn-pxn))
        if  abs(m) > 0.5:
            cv2.line(img, (pxn, pyn), (cxn, cyn), color, thickness)

   
    
    right_slope=(cyp-pyp)/(cxp-pxp)
    left_slope= (cyn-pyn)/(cxn-pxn)

    cyp=height
    pyn=height

    cxp=(cyp-pyp)/right_slope + pxp
    pxn=(pyn-cyn)/left_slope + cxn
    
    print(pxp, pyp, cxp, cyp)

    pointfill = np.array([[pxp, pyp], [cxp, cyp],[pxn, pyn],[cxn, cyn]])
    cv2.fillPoly(img, np.int32([pointfill]), color=(255, 0, 0))
    

    
    #cv2.line(img,  (cxp, height-box_height), (pxn, height-box_height), color, thickness-1)

  
    #plt.plot((pxp, cxp), (pyp, cyp), 'r')
    #plt.plot((pxn, cxn), (pyn, cyn), 'g')
    #plt.show()

def Lane_Finding_Pipeline_image(image):
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = image.shape
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[(0,ysize),(2.4*xsize/5, 1.22*ysize/2), (2.6*xsize/5, 1.22*ysize/2), (xsize,ysize)]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    rho = 2 
    theta = np.pi/180 
    threshold = 15   
    min_line_len = 40 
    max_line_gap = 200    
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    
    result = weighted_img(line_image, image, α=0.8, β=1., γ=0.)

    return result


#main modules starts:

import os

image_path = 'lane detection/test_images/solidWhiteCurve.jpg'
#image_path = 'test_images/solidWhiteRight.jpg'

image = mpimg.imread(image_path)



lines_edges = Lane_Finding_Pipeline_image(image)


img_rgb = cv2.cvtColor(lines_edges, cv2.COLOR_RGB2BGR)

cv2.imshow("out",img_rgb)  
cv2.waitKey(0)
plt.imshow(lines_edges) 
plt.show()




 

#for video:
video_cap= cv2.VideoCapture("lane detection/test_videos/leftside.mp4")
while(video_cap.isOpened()):
    _, image =video_cap.read()
    lines_edges = Lane_Finding_Pipeline_image(image)
    
    cv2.imshow("out",lines_edges)  
    
    key= cv2.waitKey(33)
  

    if key == 27:
        cv2.destroyAllWindows()
        break



