import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from image_processing_pipeline import process_image

import os

"""
    Sample Code Detect Driving Lanes from an Image
    In this code we have used the images stored in test_images folder
    and have generated the output
"""

# Fetching the sample images from the test_images directory
images = os.listdir("test_images/")

# Pipeline of tasks to process the images and detect the driving lanes
for image in images:
    img = mpimg.imread("test_images/" + image)

    fig, axarr = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle(image)
    axarr[0].imshow(img)
    axarr[0].set_title('Original Image')

    output = process_image(img)
    axarr[1].imshow(output)
    axarr[1].set_title('Processed Image')

    # Saving image
    plt.imsave('image_result/result_' + image, output)
