# Dandelion picking legged robot computer vision code

import cv2
import pyrealsense2 as realsense  # Intel RealSense cross-platform open-source API
import numpy
from imutils import grab_contours
from math import pi
import json
print("Environment ready")
import pdb

# Setup
pipeline = realsense.pipeline()
config = realsense.config()
pipeline_wrapper = realsense.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(realsense.camera_info.product_line))

config.enable_stream(realsense.stream.depth, 640, 480, realsense.format.z16, 30)  # enables depth stream
config.enable_stream(realsense.stream.color, 640, 480, realsense.format.bgr8, 30)  # enables RGB stream
# config.enable_stream(realsense.stream.color, 960, 540, realsense.format.bgr8, 30)  # enables RGB stream
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_scale = 0.0002500000118743628
print("Depth Scale is: " , depth_scale)

# Creating an align object
align_to = realsense.stream.color  # Using "align_to" to align RGB and depth frames
align = realsense.align(align_to)

# Skipping the first 10 frames to allow auto-exposure to get acclimated
for i in range(50):
    pipeline.wait_for_frames()

# Depth data text attributes
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 460)
corner1 = (10, 400)
corner2 = (10, 430)
fontScale = 0.7
fontColor = (255, 255, 255)
lineType = 1


def PICKING_ZONE(cx, cy):
    # check if dandelion centroid coordinates lie within picking zone bounding box.
    l_az = 230
    u_az = 410
    l_el = 0
    u_el = 430
    if l_az < cx < u_az and l_el < cy < u_el:
        return True
    else:
        return False

# Yellow balloon in lab
low_yellow = numpy.array([20, 195, 84])
high_yellow = numpy.array([179, 255, 255])
# low_yellow = numpy.array([16, 215, 188])
# high_yellow = numpy.array([179,255,255])

# [[20, 195, 84], [179, 255, 255]] 
kernel = numpy.ones((5,5), numpy.uint8)

try:
    while True:
        # Storing frameset for processing:
        frameset = pipeline.wait_for_frames()
        # Align depth frame to RGB frame
        aligned_frames = align.process(frameset)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validating both frames
        if not aligned_depth_frame or not color_frame:
            continue

        # color_image = numpy.asanyarray(color_frame.get_data())
        color_image1 = numpy.asanyarray(color_frame.get_data())
        depth_image = numpy.asanyarray(aligned_depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # here: apply dilation first, then erosion to color_image
        img_dilation = cv2.dilate(color_image1, kernel, iterations=1)
        color_image = cv2.erode(img_dilation, kernel, iterations=1) # with erosion applied after dilation

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
        # yellow = cv2.bitwise_and(color_image, color_image, mask=yellow_mask)

        yw_contours = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yw_contours = grab_contours(yw_contours)

        for contour in yw_contours:

            area = cv2.contourArea(contour)
            if area < 150:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity  = 4 * pi * (area / (perimeter * perimeter))
            if 0.4 < circularity < 1.45:
                cv2.drawContours(color_image,[contour],-1,(0,255,0), 2)

                (x, y, w, h) = cv2.boundingRect(contour)
                # print(x, y, w, h)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Finding centroid of object
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(color_image, (cx, cy), 3, (0, 0, 255), -1)  # cx, cy are centroid coordinates
                # cv2.rectangle(color_image, (230, 0), (410, 430), (255, 0, 0), 2)

                # Finding azimuth and elevation in degrees; FOV is 55 deg H x 42 deg V
                azimuth = round(((cx / 320) - 1) * 27, 2)
                elevation = round((1 - (cy / 240)) * 20, 2)

                # degree_sign = u'\N{DEGREE SIGN}'
                az_text = "Azimuth: " + str("{0:.2f}").format(azimuth)
                el_text = "Elevation: " + str("{0:.2f}").format(elevation)
                cv2.putText(color_image, az_text, corner1, font, fontScale, fontColor, lineType)
                cv2.putText(color_image, el_text, corner2, font, fontScale, fontColor, lineType)

                crop_depth = depth_image[x:x + w, y:y + h].astype(float)
                # print(crop_depth)

                if crop_depth.size == 0:
                    continue
                depth_res = crop_depth[crop_depth != 0]
                # print(depth_res)
                depth_measurement = depth_res * depth_scale

                if depth_measurement.size == 0:
                    continue
                # print(depth_measurement)

                dist, _, _, _ = cv2.mean(depth_measurement)
                # print(dist)
                text = "Depth: " + str("{0:.3f}m").format(dist)
                cv2.putText(color_image, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        cv2.imshow('Yellow blob', color_image)
        cv2.imshow('Depth', depth_colormap)

        key = cv2.waitKey(5)
        if key == 27:  # ASCII code for ESC key
            break

finally:
    pipeline.stop()  # releases the streams
