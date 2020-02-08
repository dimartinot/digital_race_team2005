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
from classes.DetectObstacle import *



stream = True

detect = DetectLane()
sign_detect = DetectSign()
inters_detect = DetectIntersection()
car = CarControl()
obstacle_detect = DetectObstacle()
skipFrame = 1

trackbar_created = False
depth_created = False

bridge = CvBridge()

def imageCallback(msg):

    global trackbar_created
    global count

    if trackbar_created == False:

        #cv2.namedWindow("Threshold")
        #cv2.namedWindow("View")
        #cv2.namedWindow("Binary")
        #cv2.namedWindow("Bird View")
        #cv2.namedWindow("Lane Detect")

        detect.createTrackbars()
        #sign_detect.createTrackbars()
        trackbar_created = True
        count = 0
    try:
        count+=1
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #with_lanes = find_street_lanes(cv_image)

        sign_direction = sign_detect.main(cv_image)
        decision = inters_detect.main(cv_image, sign_direction)
        #if count % 10 == 0:
        #    cv2.imshow("View", cv_image)
        #    cv2.waitKey(1)
        detect.update(cv_image, count)
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

        #print("FIN FRAME {}".format(count))


    except CvBridgeError as e:
        print("Could not convert to bgr8: {}".format(e))


def videoProcess(msg):
    global depth_created
    
    if depth_created == False:

        cv2.namedWindow("Depth")
        depth_created = True

    else:

        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow("Depth", cv_image)
        cv2.waitKey(1)
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        obstacle = obstacle_detect.main(gray)

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
    obstacle_detect = DetectObstacle()

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
