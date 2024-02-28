import cv2
import os

vidcap = cv2.VideoCapture('collect-1430597.webm')
success,image = vidcap.read()
count = 0
folder_name = "./videos"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
while success:
    if count % 4 == 0:
        cv2.imwrite(os.path.join(folder_name, "frame%d.jpg" % count), image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
