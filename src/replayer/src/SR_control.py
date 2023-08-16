# LATEST VERSION OF WORKING CODE WITH RGB STREAM ONLY
from joy.decl import *
from joy import JoyApp, progress, DEBUG
from joy.plans import FunctionCyclePlan, Plan, StickFilter, SheetPlan
from numpy import (asfarray, clip, linspace, concatenate, ones, zeros, interp,
                   newaxis, asanyarray, array, arange)
from time import time as now, sleep

import cv2
import pyrealsense2 as realsense  # Intel RealSense cross-platform open-source API
from imutils import grab_contours
import math
import numpy as np

print("Environment ready")
#  export PYGIXSCHD=pygame

SERVO_NAMES = {
    0x04: 'MR', 0x08: 'FL', 0x0C: 'HL',
    0x02: 'FR', 0x06: 'HR', 0x0A: 'ML'
}


class Buehler(object):
    def __init__(self, offset, sweep, duty):
        """
    Parameters:
      offset (angle in cycles (1.0 is full turn))
      sweep  (angle in cycles (1.0 if full turn)
      duty   (ratio from 0 to 1)
    """
        self.offset = offset
        self.sweep = sweep
        self.duty = duty
        self.resOfs = 0
        self.resOfs, _ = self.at(0)

    def at(self, phi):
        """
    Implements Buehler clock and returns phase as a function of time.
      INPUTS:
        phi -- phase in units of period 0.0 to 1.0
      OUTPUT: pos, spd
        pos -- output angle in rotations (0.0 to 1.0)
        spd -- speed, in rotations per period 
      NOTE: 0 is always pinned to producing an output of 0
    """
        phi = (phi + 1 - self.offset) % 1.0
        if phi < self.duty:
            spd = self.sweep / self.duty
            pos = phi * spd
        else:
            spd = (1 - self.sweep) / (1 - self.duty)
            pos = (phi - self.duty) * spd + self.sweep
        return (pos + 1 - self.resOfs) % 1.0, spd


def DANDELION_PICKING_MOTION(step):
    swoop_sheet = asfarray([  # [t, FR, MR, HR, FL, ML, HL]
        [0.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [1.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
        [2.5, 0.45, 0.5, 0.88, -0.45, -0.5, -0.88],
        [3.4, 0.45, 0.5, 0.90, -0.45, -0.5, -0.90],
        [4.0, 0.38, 0.5, 0.95, -0.38, -0.5, -0.95],
        [4.1, 0.33, 0.5, 0.05, -0.33, -0.5, -0.05],
        [4.2, 0.30, 0.6, 0.15, -0.30, -0.6, -0.15],
        [4.3, 0.26, 0.75, 0.18, -0.26, -0.75, -0.18],
        [4.5, 0.26, 0.8, 0.22, -0.26, -0.8, -0.22],
        [4.7, 0.26, 0.95, 0.35, -0.26, -0.95, -0.35],
        [4.8, 0.26, 1.0, 0.41, -0.26, -1.0, -0.41],
        [4.9, 0.6, 1.0, 0.41, -0.6, -1.0, -0.41],
        [5.0, 0.6, 1.0, 0.55, -0.6, -1.0, -0.55],
        [5.1, 0.6, 1.0, 0.65, -0.6, -1.0, -0.65],
        [5.2, 0.8, 0.1, 0.70, -0.8, -0.1, -0.70]])
    t0 = swoop_sheet[:, 0]
    t = linspace(t0[0], t0[-1], step)
    s = asfarray([t] + [(interp(t, t0, i) + 0.5) % 1 * 36000 for i in swoop_sheet.T[1:]])
    return [['t', 'FL', 'ML', 'HL', 'FR', 'MR', 'HR']] + [list(si) for si in s.T]


def _TURN_IN_PLACE_SHEET(turn, bias=0.075):
    ## Create sheet for Turn In Place
    ## Note: Need to take care of the scale, offset and orientation inside the sheet
    g0 = -0.3 * asfarray([1, 0, -1, -1, 0, 1]) + asfarray([0, -bias, 0, 0, bias, 0])
    t0 = asfarray([0, -turn * 0.5, 0, 0, -turn * 0.5, 0])
    g1 = g0 + t0
    g2 = g1 + t0
    g3 = asfarray([0, -turn - bias, 0, 0, -turn + bias, 0])
    # print(g0,'\n',g1,'\n',g2,'\n',g3)
    return [
        ['t', 'FL', 'ML', 'HL', 'FR', 'MR', 'HR'],
        [0.0] + list((g0 + 0.5) * 36000),
        [0.3] + list((g1 + 0.5) * 36000),
        [0.5] + list((g2 + 0.5) * 36000),
        [0.7] + list((g3 + 0.5) * 36000),
        [1.0] + list([18000] * 6),
    ]


class Walk(Plan):
    def behavior(self):
        app = self.app
        # Get legs ready for walking the first time
        progress("(say) preparing to walk")
        app.fcp.resetPhase()
        app.fcp_fun(0)
        yield self.forDuration(2)
        yield app.fcp


class Stand(Plan):
    def behavior(self):
        app = self.app
        # Need to break self.app encapsulation to get some context
        if app.walk.isRunning():
            progress("(say) halting")
            app.halt = True
            while app.walk.isRunning():
                progress('...wait...' + str(app.moving))
                yield self.forDuration(0.1)
        progress("(say) standing")
        for l in app.leg:
            l.goto(0, 0.2)
        yield self.forDuration(0.5)


class Wrap:
    def __init__(self, servo, scl, ofs=0):
        servo.set_mode("CONT")
        self.servo = servo
        self.scl = scl * 36000
        self.ofs = ofs
        self.speed = None

    def goto(self, pos, speed):
        """
        Set position and speed together
        """
        speed = abs(speed)
        if self.speed != speed:
            self.servo.set_speed(speed * 60)
            self.speed = speed
        raw = (pos + self.ofs) * self.scl
        self.servo.set_pos(raw)
        # progress(f"RAW: {self.servo.name} to POS {raw} at SPEED {speed*60}")
        return raw


#  --------------------------------------------------------------
#  --------------------------------------------------------------
# Yellow balloon in lab
low_yellow = np.array([19, 105, 122])
high_yellow = np.array([33, 255, 255])

# Depth data text attributes
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 460)
corner1 = (10, 400)
corner2 = (10, 430)
fontScale = 0.7
fontColor = (255, 255, 255)
lineType = 1

class CVPlan(Plan):
    def __init__(self, app):
        Plan.__init__(self, app)
        self.azimuth, self.elevation, self.dist, self.cx, self.cy = 0, 0, 0, 0, 0

    def behavior(self):  # TARGET_TRACKER
        # This should be the TARGET TRACKER method that returns [azimuth, elevation, range] when prompted
        self.pipeline = realsense.pipeline()
        self.config = realsense.config()
        pipeline_wrapper = realsense.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(realsense.camera_info.product_line))

        self.config.enable_stream(realsense.stream.depth, 640, 480, realsense.format.z16, 30)  # enables depth stream
        self.config.enable_stream(realsense.stream.color, 960, 540, realsense.format.bgr8, 30)  # enables RGB stream
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        # self.depth_scale = 0.0002500000118743628
        # print("Depth Scale is: ", self.depth_scale)

        # Creating an align object
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


        fpsLimit = self.app.onceEvery(0.2)
        while True:
            yield
            ###! progress('CHECKPOINT: start of while loop') ###!
            t0 = now()
            frameset = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frameset)  # Align depth frame to RGB frame

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            ###! progress('>>> wait_for_frames at '+str(now()-t0)) ###!

            # Depth data text attributes
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 460)
            corner1 = (10, 400)
            corner2 = (10, 430)
            fontScale = 0.7
            fontColor = (255, 255, 255)
            lineType = 1

            if not aligned_depth_frame or not color_frame:  # Validating both frames
                continue
            ###! progress('>>> color_frame at '+str(now()-t0)) ###!

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())


            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            low_yellow = np.array([19, 105, 122])
            high_yellow = np.array([33, 255, 255])

            yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)

            yw_contours = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            yw_contours = grab_contours(yw_contours)

            min_w, min_h = 15, 15

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
                cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)

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
                cv2.imshow('Yellow blob', color_image)
                cv2.waitKey(1)  #1 frame every 1 ms

            # color_image = asanyarray(color_frame.get_data())
            # depth_image = asanyarray(aligned_depth_frame.get_data())
            # hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            

            # yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
            # yw_contours = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # yw_contours = grab_contours(yw_contours)

            # for contour in yw_contours:
            #     yield
            #     ###! progress('>>> contours at ' + str(now() - t0)) ###!
            #     area = cv2.contourArea(contour)

            #     if area < 150:
            #         continue

            #     perimeter = cv2.arcLength(contour, True)
            #     if perimeter == 0:  # to avoid ZeroDivisionError
            #         continue

            #     circularity = 4 * pi * (area / (perimeter * perimeter))
            #     if 0 < circularity:
            #         (x, y, w, h) = cv2.boundingRect(contour)
            #         # Finding centroid of object
            #         M = cv2.moments(contour)
            #         self.cx = int(M["m10"] / M["m00"])  # cx, cy are centroid coordinates
            #         self.cy = int(M["m01"] / M["m00"])
            #         # Finding azimuth and elevation in degrees; FOV is 54 deg H x 40 deg V
            #         self.azimuth = round(((self.cx / 320) - 1) * 27, 1)
            #         self.elevation = round((1 - (self.cy / 240)) * 20, 1)
            #         # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111")

            #         crop_depth = depth_image[x:x + w, y:y + h].astype(float)

            #         if crop_depth.size == 0:
            #             continue

            #         depth_res = crop_depth[crop_depth != 0]
            #         depth_measurement = depth_res * self.depth_scale

            #         if depth_measurement.size == 0:
            #             continue

            #         self.dist, _, _, _ = cv2.mean(depth_measurement)

            #         # if fpsLimit():
            #         #cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
            #         #(x, y, w, h) = cv2.boundingRect(contour)
            #         #cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #         cv2.circle(color_image, (self.cx, self.cy), 3, (0, 0, 255), -1)
            #         az_text = "Azimuth: " + str("{0:.2f}").format(self.azimuth)
            #         #el_text = "Elevation: " + str("{0:.2f}").format(self.elevation)
            #         #text = "Depth: " + str("{0:.3f}m").format(self.dist)
            #         cv2.putText(color_image, az_text, corner1, font, fontScale, fontColor, lineType)
            #         #cv2.putText(color_image, el_text, corner2, font, fontScale, fontColor, lineType)
            #         #cv2.putText(color_image, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            #         cv2.imshow('Yellow blob', color_image)
            #         cv2.waitKey(1)  #1 frame every 1 ms

#  ----------------------------------
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


#  ----------------------------------
def TURN_STEERING(azimuth):
    """determines turn value based on azimuth in azimuth range (-27.0 deg to 27.0 deg) divided into six zones
    | -zoneL3- | -zoneL2- | -zoneL1- | -zoneR1- | -zoneR2- | -zoneR3- |"""
    zoneL3 = [round(i, 2) for i in arange(-27.0, -18.0, 0.1).tolist()]
    zoneL2 = [round(i, 2) for i in arange(-17.9, -9.0, 0.1).tolist()]
    zoneL1 = [round(i, 2) for i in arange(-8.9, 0.0, 0.1).tolist()]
    zoneR1 = [round(i, 2) for i in arange(0.0, 8.9, 0.1).tolist()]
    zoneR2 = [round(i, 2) for i in arange(9.0, 17.9, 0.1).tolist()]
    zoneR3 = [round(i, 2) for i in arange(18.0, 27.0, 0.1).tolist()]

    turn_val = 0.00
    if azimuth in zoneL3:
        # turn in place once to the left
        turn_val = 0.18
    elif azimuth in zoneR3:
        # turn in place once to the right
        turn_val = -0.18
    elif azimuth in zoneL2:
        turn_val = 0.10
    elif azimuth in zoneR2:
        turn_val = -0.10
    elif azimuth in zoneL1:
        turn_val = 0.04
    elif azimuth in zoneR1:
        turn_val = -0.04

    return turn_val


# -----------------------------------
class STEER_PLAN(Plan):
    def __init__(self):
        self.ddcvplan = CVPlan()
        self.ddcvplan.start()

    def behavior(self):
        app = self.app
        turnval = TURN_STEERING(self.ddcvplan.azimuth)
        self.turn = clip(self.turn + turnval, -0.3, 0.3)
        progress('Turn is %.2f' % self.turn)
        yield self.forDuration(2)


# -----------------------------------
class SCMHexApp(JoyApp):
    def __init__(self, *arg, **kw):
        JoyApp.__init__(self, *arg, **kw)
        # self.bueh = Buehler(-0.5, 0.8, 0.5 ) #highest stable freq is 0.48Hz
        # self.bueh = Buehler(-0.5, 0.8, 0.4 ) #highest stable freq is 0.36Hz
        # self.bueh = Buehler(-0.5, 0.8, 0.8 ) # neutralize Buehler clock
        # self.bueh = Buehler(-0.375, 0.6, 0.4 ) #highest stable freq is 0.36Hz
        self.bueh = Buehler(-0.05, 0.2, .5)

    def onStart(self):
        global off
        off = self.robot.off
        # DEBUG.extend(list('Fr'))
        self.T0 = self.now
        self.leg = [
            Wrap(self.robot.at.FL, -1, ofs=0.5), Wrap(self.robot.at.ML, -1, ofs=0.5),
            Wrap(self.robot.at.HL, -1, ofs=0.5),
            Wrap(self.robot.at.FR, 1, ofs=0.5), Wrap(self.robot.at.MR, 1, ofs=0.5), Wrap(self.robot.at.HR, 1, ofs=0.5)
        ]
        self.fcp = FunctionCyclePlan(self, self.fcp_fun, 128, maxFreq=1.75, interval=0.05)
        self.walk = Walk(self)
        self.stand = Stand(self)
        self.swoomot = SheetPlan(self, DANDELION_PICKING_MOTION(100))
        self.tipL = SheetPlan(self, _TURN_IN_PLACE_SHEET(0.2))
        self.tipR = SheetPlan(self, _TURN_IN_PLACE_SHEET(-0.2))
        self.freq = 5 / 60.0
        self.turn = 0
        self.moving = ones(6)
        self.halt = 1
        self.fcp.setPeriod(1 / self.freq)
        self.stand.start()
        self.ddcvplan = CVPlan(self)
        #self.steerplan = STEER_PLAN()

    def onStop(self):
        progress("%s --- shutting down" % str(self))

    def fcp_fun(self, phase):
        # Legs array is FL ML HL FR MR HR
        # Compute desired leg phase
        pDes = (asfarray([0, 0.5, 0, 0.5, 0, 0.5])
                + phase
                + self.turn * asfarray([1, 1, 1, -1, -1, -1]))
        if not self.halt:
            self.moving[:] = 1
        else:
            # elements close to zero angle stop moving
            self.moving[abs((pDes + .1) % 1.0) < .2] = 0
            if not any(self.moving):
                progress("(say) stopped")
                self.fcp.stop()
        # Move all the legs
        rpt = []
        for k, l in enumerate(self.leg):
            # If leg isn't moving --> skip
            if not self.moving[k]:
                rpt.append("-STOP-")
                continue
            # Use Buehler object to compute angle and speed
            pos, spd = self.bueh.at(pDes[k])
            l.goto(pos, spd / self.fcp.period)
            rpt.append("%4.2f" % pos)
        # progress("At " + " ".join(rpt))

    def _updateFreq(self, df):
        f = self.freq + df
        f = clip(f, -1.5, 1.5)
        if abs(f) < 0.1:  # too slow is "stopped"
            self.fcp.setPeriod(0)
            progress('(say) stop')
        else:
            self.fcp.setPeriod(1 / f)
            if f > 0:
                progress('(say) forward')
            else:
                progress('(say) back')
        self.freq = f
        progress('Period changed to %g, %.2f Hz' % (self.fcp.period, f))

    def _notStopped(self):
        if self.fcp.isRunning():
            self.stand.start()
            progress("(say) first stopping...")
            return True
        return False

    def remapToKey(self, evt):  ## Maps other events to keys
        if evt.type == JOYAXISMOTION:
            if evt.axis == 0:
                if evt.value < -0.5:
                    return K_LEFT
                elif evt.value > 0.5:
                    return K_RIGHT
            elif evt.axis == 1:
                if evt.value < -0.5:
                    return K_UP
                elif evt.value > 0.5:
                    return K_DOWN
        elif evt.type == JOYBUTTONDOWN:
            return {
                5: K_q,
                4: K_SPACE,
                0: K_h,
                2: K_g,
                3: K_z,
                1: K_x,
                12: K_UP,
                14: K_DOWN,
                13: K_LEFT,
                15: K_RIGHT,
                6: K_j,
                7: K_k}.get(evt.button, None)
        else:
            return None

    def onEvent(self, evt):
        if self.now - self.T0 > 3000:  # Controller time limit
            self.stop()
        return JoyApp.onEvent(self, evt)

    def on_K_SPACE(self, evt):
        "[space] stops cycles"
        self.fcp.setPeriod(0)
        self.turn = 0
        progress('Period changed to %s' % str(self.fcp.period))

    def on_K_m(self, evt):
        progress("Performing dandelion picking swooping motion")
        for leg in self.leg:
            leg.goto(0, 0.27)
        yield self.swoomot.forDuration(.5)
        if self._notStopped(): return
        self.swoomot.start()
        print('CHECKPOINT: swooping motion done')

    def on_K_h(self, evt):
        self.halt = (self.halt == 0)
        if self.halt:
            if self.walk.isRunning():
                self.walk.stop()
            # self.stand.start()
        else:
            if self.stand.isRunning():
                self.stand.stop()
            self.walk.start()

    def on_K_s(self, evt):
        if self._notStopped(): return
        progress("(say) Enter position commands")
        # Get a list from the use and complete it to 6 entries by appending zeros
        self.cmd = list(input("Positions (as 6-tuple, or [] to return to normal mode):"))
        if self.cmd:
            progress("Set to:" + str(self.cmd))
            for k, l in enumerate(self.leg):
                l.goto(self.cmd[k], 5)

    def on_K_x(self, evt):
        if self._notStopped(): return
        self.tipR.start()
        progress("(say) facing right")

    def on_K_z(self, evt):
        if self._notStopped(): return
        self.tipL.start()
        progress("(say) facing left")

    def on_K_UP(self, evt):
        return self._updateFreq(0.02)

    def on_K_DOWN(self, evt):
        return self._updateFreq(-0.02)

    def on_K_LEFT(self, evt):
        self.turn = clip(self.turn + 0.02, -0.3, 0.3)
        progress('Turn is %.2f' % self.turn)

    def on_K_RIGHT(self, evt):
        self.turn = clip(self.turn - 0.02, -0.3, 0.3)
        progress('Turn is %.2f' % self.turn)

    def on_K_j(self, evt):
        if self._notStopped(): return
        self.fcp.moveToPhase(self.fcp.phase - 0.01)

    def on_K_k(self, evt):
        if self._notStopped(): return
        self.fcp.moveToPhase(self.fcp.phase + 0.01)

    #  -----------------------------
    def MISSED_DANDELION(self, azimuth, elevation, dist, cx, cy):
        """BigANT steers backwards if dandelion is missed"""
        print('CHECKPOINT: MISSED_DANDELION method')
        if azimuth == 0 and elevation == 0 and dist == 0 and cx == 0 and cy == 0:  # all are 0 upon initialization
            return

        turnval = TURN_STEERING(azimuth)

        trigger = True
        freq = -0.16  # for steering backwards
        sleep(1.5)

        t1 = now()
        while now() < t1 + 4:  # steers ~1.5m backwards. 1.5 * 3 * (1/0.16) = 28.125 s
            if trigger is True:
                self._updateFreq(freq)
                self.turn = clip(self.turn + turnval, -0.3, 0.3)
                self.walk.start()
            trigger = False
            print('az: {}'.format(azimuth))
            # sleep(2.0)  # sleep suspends execution -- remove

        self.walk.stop()
        #  use TARGET_TRACKER (behavior method in CVPlan) and obtain [azimuth, elevation, depth] data,
        #  then continue to TARGET_FOLLOWER

    #  -----------------------------
    def TARGET_FOLLOWER(self, azimuth, elevation, dist, updatefreq, cx, cy):
        #  @ f = 0.16Hz, 1.0m distance takes ~3 Buehler clock cycles; 'x'm : 18.75x seconds; period = 1/f
        print('CHECKPOINT 3: starting TARGET_FOLLOWER method')
        if azimuth == 0 and elevation == 0 and dist == 0 and cx == 0 and cy == 0:  # all are 0 upon initialization
            print('ALL ZERO!')
            return

        time_to_dand = 18.75 * dist
        t0 = now()
        trigger = True
        turnval = TURN_STEERING(azimuth)

        # executes commands until required Buehler clock cycles are complete
        while now() < t0 + 4:  # replace 4 with time_to_dand
            #  should walk for the 'time_to_dand' time only; sets freq and walk.start once
            if trigger is True:
                self._updateFreq(updatefreq)
                self.turn = clip(self.turn + turnval, -0.3, 0.3)
                progress('Turn is %.2f' % self.turn)
                self.walk.start()

            trigger = False
            print('az: {}, el: {}, di: {}'.format(azimuth, elevation, dist))
            # sleep(2.0)  # but sleep suspends execution

        self.walk.stop()  # stops to check dandelion's position w.r.t cutting appendage

        yield  # goes to RGBD frames to check new dandelion position, then decides to pick or to not pick

        if PICKING_ZONE(cx, cy):
            print('dandelion IN PICKING ZONE')
            progress("Performing dandelion picking swooping motion")
            for leg in self.leg:
                leg.goto(0, 0.27)
            self.swoomot.forDuration(.5)  # yield removed here as it reverts to CVPlan process
            if self._notStopped(): return
            self.swoomot.start()
            # sleep(7)  # time needed to finish swooping motion before going to yield; but sleep suspends execution
            print('CHECKPOINT: swooping motion done')

        else:
            print('dandelion NOT IN PICKING ZONE')
            self.MISSED_DANDELION(self.ddcvplan.azimuth, self.ddcvplan.elevation, self.ddcvplan.dist, self.ddcvplan.cx,
                                  self.ddcvplan.cy)

        return
        # yield

    #  -----------------------------------------------
    def on_K_r(self, evt):
        print('CHECKPOINT 1: starting dandelion CV window')
        self.ddcvplan.start()
        print('CHECKPOINT 2:checking azimuth, elevation, depth:', self.ddcvplan.azimuth, self.ddcvplan.elevation,
              self.ddcvplan.dist)
    
        yield self.TARGET_FOLLOWER(self.ddcvplan.azimuth, self.ddcvplan.elevation, self.ddcvplan.dist, 0.00,
                                   self.ddcvplan.cx,
                                   self.ddcvplan.cy)  # need f = 0.16Hz
        print('CHECKPOINT: post TARGET_FOLLOWER')

    #  -----------------------------------------------
    # def on_K_c(self, evt):
    #     self.steerplan.start()


if __name__ == '__main__':
    print("""
  SCM Hexapod controller
  ----------------------

  When any key is pressed, starts a FunctionCyclePlan that runs the SCM
  hexapod. Usually works in position mode, but crosses the dead zone by
  switching between position and speed control, giving approximate RPM
  command, and polling for the motor to be out the other side

  h -- reset stance
  q -- quit
  arrow keys -- speed/slow and turn
  SPACE -- pause / resume
  z,x -- turn in place
  any other key -- start

  The application can be terminated with 'q' or [esc]
  """)

    import ckbot.dynamixel as DX
    import ckbot.logical as L
    # import ckbot.nobus as NB
    import joy

    # import joy.mxr
    # joy.mxr.DEBUG[:] = ['r']

    if 1:  # 1 to run code; 0 for simulation
        # for actual operation use w/ arch=DX & NO required line:
        L.DEFAULT_BUS = DX
        app = SCMHexApp(
            # cfg = dict( logFile = "/tmp/log", logProgress=True ),
            robot=dict(arch=DX, count=6, names=SERVO_NAMES,
                       #                       port=dict(TYPE='TTY', glob="/dev/ttyACM*", baudrate=115200)
                       port=dict(TYPE='TTY', glob="/dev/ttyUSB*", baudrate=115200)
                       ))

    joy.DEBUG[:] = []
    app.run()

'''
# debug code plotting
import joy
l = asfarray( [ (e['TIME'], e['ang'], e['nid']) for e in joy.loggit.iterlog('/tmp/log.gz') if e['TOPIC'] == "set_ang" ] )
g = asfarray( [ (e['TIME'], e['ang'], e['nid']) for e in joy.loggit.iterlog('/tmp/log.gz') if e['TOPIC'] == "get_ang" ] )
for k in set(unique(l[:,2])):
    plot( g[g[:,2]==k,0], g[g[:,2]==k,1], '+' )
    plot( l[l[:,2]==k,0], l[l[:,2]==k,1], '.' )
'''
