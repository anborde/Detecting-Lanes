from helper_functions import *
from moviepy.editor import VideoFileClip
import cv2


# Function used to process an individual image part of a video
def process_image(image):
    # Getting size metrics of the image
    imshape = image.shape

    # Step 1: Detecting Edges using Canny Edge Detection
    grayImg = grayscale(image)
    kernel_size = 5
    grayImg = cv2.GaussianBlur(grayImg,(kernel_size, kernel_size),0)

    cannyImg = canny(grayImg, 50, 150)

    # Step 2: Identifying region on interest
    vertices = np.array([[(450, 320),(525, 320), (900, imshape[0]), (150,imshape[0])]], dtype=np.int32)

    cannyImg = region_of_interest(cannyImg, vertices)


    # Step 3: Identifying edges representing lines using Hough Transform
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_len = 5
    max_line_gap = 1
    line_image = hough_lines(cannyImg, rho, theta, threshold, min_line_len, max_line_gap)

    # Step 4: Overlapping the image with lane lines i.e. line_image over original image
    output = weighted_img(line_image, image)

    return output


# Function used to find Lanes in videos
def findLanes(input_file, output_file):
    clip1 = VideoFileClip(input_file)

    # Processing input
    clip = clip1.fl_image(process_image)

    # Writing output video
    clip.write_videofile(output_file, audio=False)
