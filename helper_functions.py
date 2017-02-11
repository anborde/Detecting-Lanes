#importing some useful packages
import numpy as np
import cv2
import math


def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel

    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """
    Applies the Canny transform
    """
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with lane lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Segmenting Lines
    left, right = segment_lines(lines, img.shape[1])

    # Get the line representing the left lane
    if len(left) > 1:
        left_lane = fit_line(left, img.shape[0], 325)

        # Drawing left lane
        draw_lines(line_img, left_lane)

    if len(right) > 1:
        # Getting the line representing the right lane
        right_lane = fit_line(right, img.shape[0], 325)

        # Drawing right lane
        draw_lines(line_img, right_lane)

    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + c
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)

# Function to segment line as left and right lane
def segment_lines(lines, img_width):
    """
        This function divides the lines obtained from hough transform to left
        and right lane points based on two parameters:
        1. Slope of the line
        2. Position of the point in the image i.e. does the point fall on left
        side of image or right side of image
    """
    # Lists to store left and right lane points
    left = []
    right = []

    # Half mark with respect to the x-axis to determine whether point lies on left or right side
    half_mark = (int)(img_width / 2)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 != 0:
                slope = (y2 - y1)/(x2 - x1)
                if slope < 0:
                    # If slope negative, consider the point as a candidate for left lane
                    if x1 < half_mark:
                        left.append([[x1, y1]])
                    if x2 < half_mark:
                        left.append([[x2, y2]])
                elif slope > 0:
                    # If slope negative, consider the point as a candidate for right lane
                    if x1 > half_mark:
                        right.append([[x1, y1]])
                    if x2 > half_mark:
                        right.append([[x2, y2]])
    return left, right

# Function to fit line for the set of given points using linear regression
def fit_line (points, y1, y2):
    '''
        This function used to fit the points segemented as lane points to a line
        resembling the lane

        The input to this function are the points and the y co-ordinate extremes
        of the region of interest i.e. the highest and lowest value of y co-ordinate
        in the vertices of the region of interest

        To fit the line Linear Regression is used where for a set of points
        {(x1, y1), (x2, y2), ..., (xn, yn)} the given line is y = mx + c
        where:

        m = summation(xi - x_mean)*(yi - y_mean)/summation(xi - x_mean)^2
        c = y_mean - m * x_mean

        i ranges from 0..n
    '''
    # Numearator and Denominator for the equation to calculate 'm' (slope)
    numerator = 0
    denominator = 0

    mean_x = 0
    mean_y = 0

    for point in points:
        for x, y in point:
            mean_x += x
            mean_y += y

    # Calculating mean of x,y co-ordinates
    mean_x = mean_x / len(points)
    mean_y = mean_y / len(points)

    # Calculating the numerator and denominator for the regression formula
    for point in points:
        for x, y in point:
            numerator += ( x - mean_x) * ( y - mean_y)
            denominator += math.pow(( x - mean_x), 2)

    # Calculating m and c of the line equation y = mx + c
    m = numerator/ denominator

    c = mean_y - ( m * mean_x)

    # Extrapolating the line as per the extremities of region of interest
    x1 = (int)((y1 - c)/m)
    x2 = (int)((y2 - c)/m)

    return [[[x1, y1, x2, y2]]]