#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from classes.DetectLane import *
from classes.CarControl import *
from classes.DetectSign import *
from classes.DetectLanev2 import *



stream = True

detect = DetectLane()
# lane_detect = DetectLanev2()
sign_detect = DetectSign()
car = CarControl()
skipFrame = 1

trackbar_created = False

bridge = CvBridge()

def imageCallback(msg):

    global trackbar_created


    if trackbar_created == False:
        
        # cv2.namedWindow("Threshold")
        cv2.namedWindow("View")
        # cv2.namedWindow("Binary")
        # cv2.namedWindow("Bird View")
        # cv2.namedWindow("Lane Detect")
        
        # detect.createTrackbars()
        # sign_detect.createTrackbars()
        # lane_detect.createTrackbars()
        trackbar_created = True

    try:
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        sign_detect.main(cv_image)
        # lane_detect.main(cv_image)
        cv2.imshow("View", cv_image)
        cv2.waitKey(1)
        detect.update(cv_image)
        
        car.driverCar(
            detect.getLeftLane(),
            detect.getRightLane(),
            50
        )
        
        
    except CvBridgeError as e:
        print("Could not convert to bgr8: {}".format(e))

def videoProcess():
    while True:
        ret, frame = cap.read()

        cv2.imshow("View", frame)
        detect.update(frame)
        # lane_detect.main(frame)
        sign_detect.main(frame)
        cv2.waitKey(30)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print("launching listener")
    rospy.init_node('image_listener', anonymous=True)

    # detect = DetectLane()
    lane_detect = DetectLanev2()
    sign_detect = DetectSign()
    car = CarControl()

    if (stream):
        print("setting subscriber")
        rospy.Subscriber("team2005/camera/rgb/compressed", CompressedImage, imageCallback)

        rospy.spin()
    else:
        videoProcess()


if __name__ == '__main__':
    listener()
    #cv2.destroyAllWindows()
