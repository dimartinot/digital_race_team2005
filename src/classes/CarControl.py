#! /usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import cv2
import rospy
import std_msgs.msg
import math

steer_publisher = None
speed_publisher = None

laneWidth = 40

minVelocity = 10
maxVelocity = 50

preError = None

kP = None
kI = None
kD = None

t_kP = None
t_kI = None
t_kD = None

carPos = None

class CarControl():

    def __init__(self):

        self.step_turn = 26
        self.max_step = 25
        self.direction = None

        self.carPos = cv2.KeyPoint(
            120,
            300,
            _size=0
        )

        self.coefRight = 3
        self.coefLeft = 2

        self.steer_publisher = rospy.Publisher("/team2005/set_angle", std_msgs.msg.Float32, queue_size=10)
        self.speed_publisher = rospy.Publisher("/team2005/set_speed", std_msgs.msg.Float32, queue_size=10)

        self.stay_left = False # indicates if the car has to stay on the left of the road

    def set_stay_left(self):
        self.stay_left = True

    def unset_stay_left(self):
        self.stay_left = False

    def errorAngle(self, dst):
        (dst_x, dst_y) = (dst[0], dst[1])
        (carPos_x, carPos_y) = self.carPos.pt

        if (dst_x == carPos_x):
            return 0
        if (dst_y == carPos_y):
            return (-90 if dst_x < carPos_x else 90)

        pi = math.acos(-1.0)
        dx = dst_x - carPos_x
        dy = carPos_y - dst_y

        if (dx < 0):
            return - math.atan(-dx/dy)*180/pi

        return math.atan(dx/dy)*180/pi

    def driverCar(self, left, right, velocity):

        if (len(left) > 11 and len(right) > 11):
            i = len(left) - 11
            error = preError

            while (left[i] == None and right[i] == None):
                i -= 1
                if (i < 0):
                    return

            if (left[i] != None and right[i] != None):
                #error = self.errorAngle((np.array(left[i]) + np.array(right[i])) / 2)
                # margin coefficients:

                if (self.stay_left):
                    error = self.errorAngle(self.coefRight*np.array(left[i]) / 5 + self.coefLeft*np.array(right[i])/5)
                else:
                    error = self.errorAngle(self.coefLeft*np.array(left[i]) / 5 + self.coefRight*np.array(right[i])/5)

            elif left[i] != None:
                error = self.errorAngle(np.array(left[i]) + np.array([laneWidth / 2, 0]))

            else:

                error = self.errorAngle(np.array(right[i]) - np.array([laneWidth / 2, 0]))

            
            if abs(error) >20:
                error = 0
                
            self.steer_publisher.publish(std_msgs.msg.Float32(error))
            self.speed_publisher.publish(std_msgs.msg.Float32(velocity))

    def goStraight(self, velocity=10):
        self.steer_publisher.publish(std_msgs.msg.Float32(0))
        self.speed_publisher.publish(std_msgs.msg.Float32(velocity))

    def turnHere(self):
        """ Exécute la maneuvre nécessaire pour un virage.
    
        ENTREE: direction: "left" to trun left or "right to turn right"
        """

        angle = 0

        turning = self.step_turn < self.max_step

        if turning:
            if self.direction == "right":
                angle = 20 # to turn right
            elif self.direction == "left":
                angle = -20 # to turn left


        self.steer_publisher.publish(std_msgs.msg.Float32(angle))

        self.step_turn+=1

        return turning

    def obstacle(self, keypoint):
        
        height = 240
        width = 320
        center = [160,120]
        dangerZone = False
        angle = 0
        distance = center[0] - keypoint.pt[0]

        if abs(distance) < 100:
            dangerZone = True

        if dangerZone == True:
            if distance > 0:
                angle = 5
            elif distance < 0:
                angle = -5
        #print('angle: {}'.format(angle))
        #self.steer_publisher.publish(std_msgs.msg.Float32(angle))
