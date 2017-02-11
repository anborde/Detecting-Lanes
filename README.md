# Detecting-Lanes
A project to detect driving lanes in image and video with the help of opencv library. The project is done as part of Udacity's Nanodegree Program

## Table of Content
1. **helper_functions.py**
  - This file consists of various image transformation functions which are used in the pipeline of tasks for detecting lanes

2. **image_processing_pipeline**
  - This file consists of chain of tasks executed to process an image and detect the lines

3. **video_processing_pipeline**
  - This file consists of functions which execute a chain of tasks to process an video and detect lanes

4. **lane_detection_image.py**
  - This is the sample code which reads images from the folder test_images and stores the result in image_result folder
  
5. **lane_detection_video.py**
  - This is the sample code which reads the sample video file and detects the driving lanes
 
## Requirements
- opencv 3.1.0
- moviepy
  
## Usage
- Use the pipeline files to process the image or video.
- The project is very naive in nature and hence is limited to the set of images and video that can be processed.
- In order to process the image, we need to first manually identify the region where the lanes are present. This region is referred as 'Region of Interest'
- We then pass the image/ video and the vertices of Region of Interest to `process_image()` in image_processing_pipeline.py for image and `find_lanes()` in video_processing_pipeline.py for video respectively.
- For video we calculate video size by using the function `getVideoSize()` in video_processing_pipeline.py

## Issues
- The application of project is to a limited set of images
- Vertices for region of interest is hardcoded
- The lanes are not perfectly detected as simple concepts of Computer Vision are used instead of concepts like Deep Learning + Computer Vision

## License
License included in the repo
