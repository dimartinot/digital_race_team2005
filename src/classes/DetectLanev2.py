import cv2
import numpy as np

skyLine = 85

class DetectLanev2():
    def __init__(self):

        self.slideThickness = 10
        self.BIRDVIEW_WIDTH = 240
        self.BIRDVIEW_HEIGHT = 320
        self.VERTICAL = 0
        self.HORIZONTAL = 1
        self.leftLane = []
        self.rightLane = []

        self.params= {
            'lB': 0,
            'lG': 0,
            'lR': 195,
            'hB': 179,
            'hG': 255,
            'hR': 255,
            'shadow': 0
        }

    
    def createTrackbars(self):
        
        cv2.namedWindow('Processed')

        cv2.createTrackbar('lB trackbar', 'Processed', self.params['lB'], 255, self.updatelB)
        cv2.createTrackbar('lG trackbar', 'Processed', self.params['lG'], 255, self.updatelG)
        cv2.createTrackbar('lR trackbar', 'Processed', self.params['lR'], 255, self.updatelR)
        cv2.createTrackbar('hB trackbar', 'Processed', self.params['hB'], 255, self.updatehB)
        cv2.createTrackbar('hG trackbar', 'Processed', self.params['hG'], 255, self.updatehG)
        cv2.createTrackbar('hR trackbar', 'Processed', self.params['hR'], 255, self.updatehR)

    def updatelB(self, value):
        self.params['lB']=value
        return

    def updatelG(self, value):
        self.params['lG']=value
        return

    def updatelR(self, value):
        self.params['lR']=value
        return

    def updatehB(self, value):
        self.params['hB']=value
        return

    def updatehG(self, value):
        self.params['hG']=value
        return

    def updatehR(self, value):
        self.params['hR']=value
        return
    
    def BIRDVIEWTranform(self, src):
        height, width = src.shape

        src_vertices = np.array([
            [0, skyLine],
            [width, skyLine],
            [width, height],
            [0, height]
        ], dtype="float32")

        dst_vertices = np.array([
            [0,0],
            [self.BIRDVIEW_WIDTH, 0],
            [self.BIRDVIEW_WIDTH - 105, self.BIRDVIEW_HEIGHT],
            [105, self.BIRDVIEW_HEIGHT]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)

        warp = cv2.warpPerspective(src, M, (self.BIRDVIEW_HEIGHT, self.BIRDVIEW_WIDTH), flags=cv2.INTER_LINEAR, borderValue = cv2.BORDER_CONSTANT)
        return warp

    def splitLayer(self, src, dire):
        """
            Split an image in multiple subimages.
            Input:
                - src: cv2 image
                - dire: direction constant
        """
        (rowN, colN) = src.shape
        res = []
        ## UNSURE ABOUT SLICING
        if (dire == self.VERTICAL):
            # range(start, stop, step)
            for i in range(0, rowN - self.slideThickness, self.slideThickness):
                # croping is much easier in Python, it is basically just slicing
                tmp = src[i:self.slideThickness, 0:colN]
                res.append(tmp)

        else:

            for i in range(0, colN - self.slideThickness, self.slideThickness):
                # croping is much easier in Python, it is basically just slicing
                tmp = src[0:self.slideThickness, i:rowN]
                res.append(tmp)

        return res
    
    def centerRoadSide(self, layers, dire):

        res = []
        inputN = len(layers)

        for i in range(inputN):
            _,contours,_ = cv2.findContours(layers[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmp = []

            contourN = len(contours)

            for j in range(contourN):
                area = cv2.contourArea(contours[j], False)

                if area > 3:
                    M1 = cv2.moments(contours[j], False)
                    center = cv2.KeyPoint(M1["m10"] / M1["m00"], M1["m01"] / M1["m00"], _size=0)
                    (x, y) = center.pt

                    if (dire == self.VERTICAL):
                        center = cv2.KeyPoint(
                            x,
                            y + self.slideThickness * i,
                            _size=0
                        )

                    else:
                        center = cv2.KeyPoint(
                            x + self.slideThickness * i,
                            y,
                            _size=0
                        )

                    (x, y) = center.pt

                    if (x > 0 and y > 0):
                        tmp.append(center)

            res.append(tmp)

        # np use is important: vector<vector<Point> > will be translated as a np 2D-array of cv2 KeyPoints
        return np.array(res)

    def detectLeftRight(self, points):
        """
        Method used to detect left and right lanes
        Input:
            - points: numpy 2D-array of cv2 KeyPoint instances
        """
        lane1 = []
        lane2 = []
        self.leftLane = [None for _ in range(int(np.floor(self.BIRDVIEW_HEIGHT / self.slideThickness)))]
        self.rightLane = [None for _ in range(int(np.floor(self.BIRDVIEW_HEIGHT / self.slideThickness)))]

        pointMap = np.zeros((points.shape[0], 20))
        prePoint = np.zeros((points.shape[0], 20))
        postPoint = np.zeros((points.shape[0], 20))

        dis = 10
        max1 = -1; max2 = -1

        ##
        ##  /!\ UNSAFE LOOP, TODO: FIX
        ##
        for i in range(points.shape[0]):
            for j in range(len(points[i])):
                pointMap[i][j] = 1
                prePoint[i][j] = -1
                postPoint[i][j] = -1

        for i in reversed(range(points.shape[0] - 2)):

            for j in range(len(points[i])):

                err = 320
                for m in range(1, min(points.shape[0] - 1 - i, 5)):
                    check = False ## TODO: why unused ?

                    for k in range(len(points[i + 1])):

                        (x_m, y_m) = points[i + m][k].pt
                        (x, y) = points[i][j].pt

                        if (abs(x_m - x) < dis and abs(y_m - y) < err):
                            err = abs(x_m - x)

                            pointMap[i][j] = pointMap[i + m][k] + 1
                            prePoint[i][j] = k
                            postPoint[i + m][k] = j
                            check = True

                    break ## breaks out of the m loop. Why is it not conditioned by check ? TODO: ???

                if (pointMap[i][j] > max1):
                    max1 = pointMap[i][j]
                    posMax = cv2.KeyPoint(i, j, _size=0)

        for i in range(points.shape[0]):
            for j in range(len(points[i])):
                if (pointMap[i][j] > max2 and (i != posMax.pt[0] or j != posMax.pt[1]) and postPoint[i][j] == -1):
                    max2 = pointMap[i][j]
                    posMax2 = cv2.KeyPoint(i, j, _size=0)



        if max1 == -1:
            return

        # DEFINES LANE 1 POINTS
        while (max1 >= 1):
            (x,y) = points[int(posMax.pt[0])][int(posMax.pt[1])].pt
            lane1.append(
                [x,y]
            )
            if (max1 == 1):
                break

            posMax = cv2.KeyPoint(
                posMax.pt[0]+1,
                prePoint[int(posMax.pt[0])][int(posMax.pt[1])],
                _size=0
            )

            max1 -= 1

        # DEFINES LANE 2 POINTS
        while (max2 >= 1):
            (x,y) = points[int(posMax2.pt[0])][int(posMax2.pt[1])].pt
            lane2.append(
                [x, y]
            )
            if (max2 == 1):
                break

            posMax2 = cv2.KeyPoint(
                posMax2.pt[0]+1,
                prePoint[int(posMax2.pt[0])][int(posMax2.pt[1])],
                _size=0
            )

            max2-= 1

        subLane1 = np.array(lane1[0:5])
        subLane2 = np.array(lane2[0:5])

        # checking if sublane has an empty value
        if len(subLane1)!=0 and len(subLane2)!=0:
            line1 = cv2.fitLine(subLane1, 2, 0, 0.01, 0.01)
            line2 = cv2.fitLine(subLane2, 2, 0, 0.01, 0.01) 

            #print(line1, line2)
            lane1X = (self.BIRDVIEW_WIDTH - line1[3]) * line1[0] / line1[1] + line1[2]
            lane2X = (self.BIRDVIEW_WIDTH - line2[3]) * line2[0] / line2[1] + line2[2]

            if (lane1X < lane2X):
                for i in range(len(lane1)):
                    self.leftLane[int(np.floor(lane1[i][1] / self.slideThickness ))] = lane1[i]

                for i in range(len(lane2)):
                    self.rightLane[int(np.floor(lane2[i][1] / self.slideThickness ))] = lane2[i]

            else:

                for i in range(len(lane1)):
                    self.rightLane[int(np.floor(lane1[i][1] / self.slideThickness ))] = lane1[i]

                for i in range(len(lane2)):
                    self.leftLane[int(np.floor(lane2[i][1] / self.slideThickness ))] = lane2[i]

    def main(self, original):
        
        frame_rgb = np.copy(original)
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
        
        lowerLimits = np.array([self.params['lB'], self.params['lG'], self.params['lR']])
        upperLimits = np.array([self.params['hB'], self.params['hG'], self.params['hR']])

        # Our operations on the frame come here
        thresholded = cv2.inRange(frame_hsv, lowerLimits, upperLimits)

        processed_thresh = self.BIRDVIEWTranform(thresholded)

        lines = cv2.HoughLinesP(processed_thresh, 1, np.pi/180, 1)
        for i in range(lines.shape[0]):
            l = lines[i]
            cv2.line(processed_thresh, (l[0][0], l[0][1]), (l[0][2], l[0][3]), 255, 3, cv2.LINE_AA)

        layers = self.splitLayer(processed_thresh, self.VERTICAL)
        points = self.centerRoadSide(layers, self.VERTICAL)

        self.detectLeftRight(points)

        lanes = np.zeros(processed_thresh.shape, dtype=np.uint8)

        lanes = cv2.cvtColor(lanes, cv2.COLOR_GRAY2BGR)
        
        for i in range(1, len(self.leftLane)):
            if (self.leftLane[i] != None):
                cv2.circle(lanes, tuple(self.leftLane[i]), 1, (0,0,255), 2, 8, 0)

        for i in range(1, len(self.rightLane)):
            if (self.rightLane[i] != None):
                cv2.circle(lanes, tuple(self.rightLane[i]), 1, (255,0,0), 2, 8, 0)
        """
        for layer in points:
            for point in layer:
                print(int(point.pt[0]))
                cv2.circle(test, tuple((int(point.pt[0]), int(point.pt[1]))), 1, (0,0,255), 2, 8, 0)
        """
        cv2.imshow('Mask', thresholded)
        cv2.imshow('Bird view', processed_thresh)
        cv2.imshow('Results', lanes)