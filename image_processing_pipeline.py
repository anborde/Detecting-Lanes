from helper_functions import *


def process_image(img, vertices):
    # Step 1: Detecting Edges using Canny Edge Detection
    grayImg = grayscale(img)

    cannyImg = canny(grayImg, 50, 150)

    # Step 2: Identifying region on interest i.e. the region where the lanes are situated
    vertices = np.array(vertices, dtype=np.int32)

    cannyImg = region_of_interest(cannyImg, vertices)

    # Step 3: Identifying edges representing lines using Hough Transform
    rho = 1
    theta = np.pi / 180
    threshold = 1
    min_line_len = 5
    max_line_gap = 1
    line_image = hough_lines(cannyImg, rho, theta, threshold, min_line_len, max_line_gap)

    # Step 4: Overlapping the image with lane lines i.e. line_image over original image
    output = weighted_img(line_image, img)

    return output
