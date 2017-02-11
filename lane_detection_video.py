from video_processing_pipeline import findLanes, getVideoSize

"""
    Sample code to detct lanes in a video using video_processing_pipeline.py
"""

# Input file
input_file = "solidWhiteRight.mp4"

# Getting Video Size
shape = getVideoSize(input_file)

# Defining the vertices for region of interest
vertices = [[(450, 320),(525, 320), (900, shape[0]), (150, shape[0])]]

findLanes(input_file, "white.mp4", vertices)