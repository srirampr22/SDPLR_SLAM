#!/usr/bin/env python3

import pyrealsense2 as realsense
import numpy as np
import cv2
from imutils import grab_contours
import json
import zmq
import math

class ImageProcessing:
    def __init__(self):
        # Create a pipeline
        self.pipeline = realsense.pipeline()
        self.config = realsense.config()
        self.bboxes = None
        pipeline_wrapper = realsense.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(realsense.camera_info.product_line))

        # Configure stream
        self.config.enable_stream(realsense.stream.depth, 640, 480, realsense.format.z16, 30)  # enables depth stream
        self.config.enable_stream(realsense.stream.color, 960, 540, realsense.format.bgr8, 30)  # enables RGB stream


        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", self.depth_scale)

        # Align the depth and color stream
        self.align_to = realsense.stream.color
        self.align = realsense.align(self.align_to)

        # ZMQ publisher
        # zmq_context = zmq.Context()
        # self.zmq_publisher = zmq_context.socket(zmq.PUB)
        # self.zmq_publisher.bind("tcp://192.168.1.100:5556")


        # Skipping the first 10 frames to allow auto-exposure to get acclimated
        for i in range(20):
            # pass
            self.pipeline.wait_for_frames()

               
    def process_frames(self):

        while True:
            print('CHECKPOINT: start of while loop')

            # Depth data text attributes
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 460)
            corner1 = (10, 400)
            corner2 = (10, 430)
            fontScale = 0.7
            fontColor = (255, 255, 255)
            lineType = 1

            frameset = self.pipeline.wait_for_frames()  # Storing frameset for processing:

            aligned_stream = self.align.process(frameset)
            aligned_depth_frame = aligned_stream.get_depth_frame()
            
            color_frame = frameset.get_color_frame()

            if not color_frame:
                yield
                # continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())


            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            low_yellow = np.array([19, 105, 122])
            high_yellow = np.array([33, 255, 255])

            yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)

            yw_contours = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            yw_contours = grab_contours(yw_contours)

            min_w, min_h = 1, 1

            self.bboxes = [cv2.boundingRect(
                        contour) for contour in yw_contours if cv2.contourArea(contour) > min_w * min_h]
            
            valid_bboxes = self.bboxes
            # dist_list = []

            # CONTOUR PROCESSING STEP
            for bbox in valid_bboxes:
                (x, y, w, h) = bbox

                # Extract the contour from the original list of contours using the bounding box coordinates
                contour = None
                for cnt in yw_contours:
                    (cx, cy, cw, ch) = cv2.boundingRect(cnt)
                    if x == cx and y == cy and w == cw and h == ch:
                        contour = cnt
                        break

                # Now process the contour as before
                area = cv2.contourArea(contour)
                if area < 150:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:  # to avoid ZeroDivisionError
                    continue
                
                pi = math.pi

                circularity = math.pow(perimeter, 2) / (4 * math.pi * area)
                # if 0.4 < circularity < 1.45:
                # cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)

                mask = np.zeros_like(color_image)

                cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)  # -1 filled the contour

                # Bitwise-AND mask and original image
                result = cv2.bitwise_and(color_image, mask)

                # split the result into R, G, B channels
                r, g, b = cv2.split(result)

                # find min and max of each channel in the result where mask is not zero
                r_min, r_max = r[r != 0].min(), r[r != 0].max()
                g_min, g_max = g[g != 0].min(), g[g != 0].max()
                b_min, b_max = b[b != 0].min(), b[b != 0].max()

                print(f"R min, max: {r_min}, {r_max}")
                print(f"G min, max: {g_min}, {g_max}")
                print(f"B min, max: {b_min}, {b_max}")

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Finding centroid of object
                M = cv2.moments(contour)
                self.cx = int(M["m10"] / M["m00"])  # cx, cy are centroid coordinates
                self.cy = int(M["m01"] / M["m00"])
                cv2.circle(color_image, (self.cx, self.cy), 3, (0, 0, 255), -1)
                # cv2.rectangle(color_image, (230, 0), (410, 430), (255, 0, 0), 2)  # dandelion picking bb

                # Finding azimuth and elevation in degrees; FOV is 54 deg H x 40 deg V
                self.azimuth = round(((self.cx / 320) - 1) * 27, 1)
                self.elevation = round((1 - (self.cy / 240)) * 20, 1)
                depth_value = depth_image[self.cy, self.cx]

                distance_meters = depth_value * self.depth_scale
                inches_per_meter = 39.3701
                inches = distance_meters * inches_per_meter
                # dist_list.appened(distance_meters)

                # degree_sign = u'\N{DEGREE SIGN}'
                az_text = "Azimuth: " + str("{0:.2f}").format(self.azimuth)
                el_text = "Elevation: " + str("{0:.2f}").format(self.elevation)
                cv2.putText(color_image, az_text, corner1, font, fontScale, fontColor, lineType)
                cv2.putText(color_image, el_text, corner2, font, fontScale, fontColor, lineType)
                distance_text = "Distance: {:.2f} inches".format(inches)
                # circularity_text = "Circularity: {:.2f} meters".format(circularity)
                cv2.putText(color_image, distance_text, (self.cx, self.cy + 30), font, fontScale, fontColor, lineType)
                # cv2.putText(color_image, circularity_text, (self.cx, self.cy + 30), font, fontScale, fontColor, lineType)


            cv2.imshow('Yellow blob', color_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            print('CHECKPOINT: after yield')

        cv2.destroyAllWindows()

    # def publish_zmq_message(self, message):
    #     json_message = json.dumps(message)
    #     self.zmq_publisher.send_string(json_message)

def main():
    image_processor = ImageProcessing()
    for _ in image_processor.process_frames():
        pass


if __name__ == "__main__":
    main()
