#!/usr/bin/env python
import os
import numpy as np
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from classes.DetectLane import *
from classes.CarControl import *
from classes.DetectSign import *
from classes.DetectIntersection import *



stream = True

detect = DetectLane()
sign_detect = DetectSign()
inters_detect = DetectIntersection()
car = CarControl()

skipFrame = 1

trackbar_created = False
depth_created = False

bridge = CvBridge()

def imageCallback(msg):

    global trackbar_created


    if trackbar_created == False:

        cv2.namedWindow("Threshold")
        cv2.namedWindow("View")
        cv2.namedWindow("Binary")
        cv2.namedWindow("Bird View")
        cv2.namedWindow("Lane Detect")

        detect.createTrackbars()
        #sign_detect.createTrackbars()
        trackbar_created = True

    try:
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        #with_lanes = find_street_lanes(cv_image)

        sign_direction = sign_detect.main(cv_image)
        decision = inters_detect.main(cv_image, sign_direction)


        cv2.imshow("View", cv_image)
        cv2.waitKey(1)
        detect.update(cv_image)

        if decision:
            car.step_turn = 0
            car.direction = decision

        if car.step_turn <= car.max_step:
            car.turnHere()
        else:
            car.driverCar(
            detect.getLeftLane(),
            detect.getRightLane(),
            40
        )


    except CvBridgeError as e:
        print("Could not convert to bgr8: {}".format(e))


def videoProcess(msg):
    global depth_created

    if depth_created == False:

        # cv2.namedWindow("Depth")
        depth_created = True

    else:

        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # cv_image[cv_image > 127] = 0

        # print('TEST:', cv_image[0])
        # circle = inters_detect.checking(cv_image)
        # print('CHECK:', circle)
        """
        if circle:
            print('circle')
        """
        # cv2.imshow("Depth", cv_image)
        # detect.update(frame)
        # sign_detect.main(frame)
        cv2.waitKey(1)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print("launching listener")
    rospy.init_node('image_listener', anonymous=True)

    # detect = DetectLane()
    sign_detect = DetectSign()
    car = CarControl()

    if (stream):
        print("setting subscriber")
        rospy.Subscriber("team2005/camera/rgb/compressed", CompressedImage, imageCallback)
        rospy.Subscriber("team2005/camera/depth/compressed", CompressedImage, videoProcess)

        rospy.spin()
    else:
        videoProcess()


if __name__ == '__main__':
    listener()
    #cv2.destroyAllWindows()
