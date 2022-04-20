import sys
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from functions import *
#
input_video = sys.argv[1]
output_video = sys.argv[2]
debug_flag = sys.argv[3]

# a = 'challenge_video.mp4'


if debug_flag == 'y':
    myclip = VideoFileClip(input_video)
    output_vid = output_video
    clip = myclip.fl_image(pipeline_deb_mode)
    clip.write_videofile(output_vid, audio=False)
else:
    myclip = VideoFileClip(input_video)
    output_vid = output_video
    clip = myclip.fl_image(pipeline_normal_mode)
    clip.write_videofile(output_vid, audio=False)



