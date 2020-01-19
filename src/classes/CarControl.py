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

        self.carPos = cv2.KeyPoint(
            120,
            300,
            _size=0
        )

        self.steer_publisher = rospy.Publisher("/team2005/set_angle", std_msgs.msg.Float32, queue_size=10)
        self.speed_publisher = rospy.Publisher("/team2005/set_speed", std_msgs.msg.Float32, queue_size=10)

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
                error = self.errorAngle((np.array(left[i]) + np.array(right[i])) / 2)

            elif left[i] != None:
                error = self.errorAngle(np.array(left[i]) + np.array([laneWidth / 2, 0]));

            else:

                error = self.errorAngle(np.array(right[i]) - np.array([laneWidth / 2, 0]))

            self.steer_publisher.publish(std_msgs.msg.Float32(error))
            self.speed_publisher.publish(std_msgs.msg.Float32(velocity))

    def goStraight(self, velocity=10):
        self.steer_publisher.publish(std_msgs.msg.Float32(0))
        self.speed_publisher.publish(std_msgs.msg.Float32(velocity))
