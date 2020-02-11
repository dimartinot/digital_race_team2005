import numpy as np
import cv2

# lines
minThreshold = np.array([35, 0, 184])
maxThreshold = np.array([179, 31, 255])

# borders
# minThreshold = np.array([0,58,226])
# maxThreshold = np.array([255,84,255])

minShadowTh = np.array([90, 43, 36])
maxShadowTh = np.array([120, 81, 171])
minLaneInShadow = np.array([90, 43, 97])
maxLaneInShadow = np.array([120, 80, 171])
binaryThreshold = 180

skyLine = 85
shadowParam = 40

skipFrame = 1


def to_pass(x):
    pass

def average_lane(memory, input_):

    tmp = memory
    tmp.append(input_)
    means = []
    for x in range(len(tmp[0])):
        res = [0,0]
        count = 0
        if input_[x] != None or memory[0][x] != None:
            for y in range(len(tmp)):
                if tmp[y][x]!=None:
                    res[0] += tmp[y][x][0]
                    res[1] += tmp[y][x][1]
                    count += 1
        
        if count>0:
            res[0]/=count
            res[1]/=count
        else: 
            res = None

        means.append(res)
    return(means)

class DetectLane():

    def __init__(self):

        self.slideThickness = 10
        self.BIRDVIEW_WIDTH = 240
        self.BIRDVIEW_HEIGHT = 320
        self.VERTICAL = 0
        self.HORIZONTAL = 1
        self.leftLane = []
        self.rightLane = []
        self.pos_obstacle = []
        self.memory_lane_left  = [[None]*32]*3
        self.memory_lane_right = [[None]*32]*3
    

    def updatelB(self, value):
        global minThreshold
        minThreshold[0]=value
        return

    def updatelG(self, value):
        global minThreshold
        minThreshold[1]=value
        return

    def updatelR(self, value):
        global minThreshold
        minThreshold[2]=value
        return

    def updatehB(self, value):
        global maxThreshold
        maxThreshold[0]=value
        return

    def updatehG(self, value):
        global maxThreshold
        maxThreshold[1]=value
        return

    def updatehR(self, value):
        global maxThreshold
        maxThreshold[2]=value
        return

    def createTrackbars(self):
        cv2.createTrackbar("LowH", "Threshold", minThreshold[0], 179, self.updatelB)
        cv2.createTrackbar("HighH", "Threshold", maxThreshold[0], 179, self.updatehB)

        cv2.createTrackbar("LowS", "Threshold", minThreshold[1], 255, self.updatelG)
        cv2.createTrackbar("HighS", "Threshold", maxThreshold[1], 255, self.updatehG)

        cv2.createTrackbar("LowV", "Threshold", minThreshold[2], 255, self.updatelR)
        cv2.createTrackbar("HighV", "Threshold", maxThreshold[2], 255, self.updatehR)

        cv2.createTrackbar("Shadow Param", "Threshold", shadowParam, 255, to_pass)

    def getLeftLane(self):
        return self.leftLane

    def getRightLane(self):
        return self.rightLane

    def update(self, src, count):
        
        img = self.preProcess(src, count)
        layers1 = self.splitLayer(img, self.VERTICAL)
        # print('LAYERS:', layers1)
        points1 = self.centerRoadSide(layers1, self.VERTICAL)

        self.detectLeftRight(points1)
        
        ####
        left_lane = self.getLeftLane()
        right_lane = self.getRightLane()

        len_left = len(np.where(np.array(left_lane) != None)[0])
        len_right = len(np.where(np.array(right_lane) != None)[0])


        # lane = np.zeros(img.shape, dtype=np.uint8)

        # lane = cv2.cvtColor(lane, cv2.COLOR_GRAY2BGR)
        
        # for i in range(1, len(self.leftLane)):
        #     if (self.leftLane[i] != None):
        #         cv2.circle(lane, (int(self.leftLane[i][0]), int(self.leftLane[i][1])), 1, (0,0,255), 2, 8, 0)

        # for i in range(1, len(self.rightLane)):
        #     if (self.rightLane[i] != None):
        #         cv2.circle(lane, (int(self.rightLane[i][0]), int(self.rightLane[i][1])), 1, (255,0,0), 2, 8, 0)

        
        # cv2.imshow("Lane Detect brut", lane)

        ### CLONE LANE
        if len_left >= len_right + 6:
            self.rightLane = [[e[0]+35, e[1]] if e != None else None for e in left_lane]

        elif len_right >= len_left + 6:
            self.leftLane = [[e[0]-35, e[1]] if e != None else None for e in right_lane]

        # lane = np.zeros(img.shape, dtype=np.uint8)

        # lane = cv2.cvtColor(lane, cv2.COLOR_GRAY2BGR)

        # for i in range(1, len(self.leftLane)):
        #     if (self.leftLane[i] != None):
        #         cv2.circle(lane, (int(self.leftLane[i][0]), int(self.leftLane[i][1])), 1, (0,0,255), 2, 8, 0)

        # for i in range(1, len(self.rightLane)):
        #     if (self.rightLane[i] != None):
        #         cv2.circle(lane, (int(self.rightLane[i][0]), int(self.rightLane[i][1])), 1, (255,0,0), 2, 8, 0)
        
        # cv2.imshow("Lane Detect pre", lane)


        ### AVERAGE LANES
        # self.rightLane = average_lane(self.memory_lane_right, self.rightLane)
        # self.leftLane = average_lane(self.memory_lane_left, self.leftLane)
        
        # self.memory_lane_left = [self.leftLane, self.memory_lane_left[0], self.memory_lane_left[1]]
        # self.memory_lane_right = [self.rightLane, self.memory_lane_right[0], self.memory_lane_right[1]]

        ####

        self.BIRDVIEW = np.zeros(img.shape, dtype=np.uint8)
        lane = np.zeros(img.shape, dtype=np.uint8)

        lane = cv2.cvtColor(lane, cv2.COLOR_GRAY2BGR)
        """
        for i in range(points1.shape[0]):
            for j in range(len(points1[i])):
                (x,y) = points1[i][j].pt
                x = int(x)
                y = int(y)

                cv2.circle(self.BIRDVIEW, (x,y) , 1, (0,0,255), 2, 8, 0)
        """
        for i in range(1, len(self.leftLane)):
            if (self.leftLane[i] != None):
                cv2.circle(lane, (int(self.leftLane[i][0]), int(self.leftLane[i][1])), 1, (0,0,255), 2, 8, 0)

        for i in range(1, len(self.rightLane)):
            if (self.rightLane[i] != None):
                cv2.circle(lane, (int(self.rightLane[i][0]), int(self.rightLane[i][1])), 1, (255,0,0), 2, 8, 0)

        
        cv2.imshow("Lane Detect post", lane)
        
    
    


    def preProcess(self, src, count):

        imgHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        
        imgThresholded = cv2.inRange(
            imgHSV,
            minThreshold[0:3],
            maxThreshold[0:3]
        )
        # imgThresholded = cv2.GaussianBlur(imgThresholded,(11,11),0)
        if self.pos_obstacle != []:
            cv2.circle(imgThresholded, (int(self.pos_obstacle.pt[0]),int(self.pos_obstacle.pt[1])+10), 17, (255,255,255),cv2.FILLED, 1)

        dst = self.BIRDVIEWTranform(imgThresholded)

        # 
        # dst[:, :40] = 0
        # dst[:, -110:] = 0
        #

        #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #if count % 10 == 0:
        
        cv2.imshow("Bird View", dst)
        #cv2.waitKey(0)
        self.fillLane(dst)
        # cv2.imshow("Binary", imgThresholded)
        #cv2.waitKey(10)

        return dst


    def laneInShadow(self, src):
        shadow = np.zeros(src.shape)

        imgHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        shadowMask = cv2.inRange(
            imgHSV,
            minShadowTh[0:3],
            maxShadowTh[0:3]
        )

        locs = np.where(shadowMask != 0)
        shadow[locs[0], locs[1]] = src[locs[0], locs[1]]

        shadowHSV = cv2.cvtColor(shadow, cv2.COLOR_BGR2HSV)
        laneShadow = cv2.inRange(
            shadowHSV,
            minLaneInShadow[0:3],
            maxLaneInShadow[0:3]
        )

        return laneShadow


    def fillLane(self, src):
        """
        Draw HoughLines on the input src image.
        Input:
            - src: cv2 image
        """
        lines = cv2.HoughLinesP(src, 1, np.pi/180, 1)
        try:
            for i in range(lines.shape[0]): #FIXME: "'NoneType' object has no attribute 'shape'" possible
                l = lines[i]
                cv2.line(src, (l[0][0], l[0][1]), (l[0][2], l[0][3]), 255, 3, cv2.LINE_AA)
        
        except:
            pass

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
                tmp = src[i:i+self.slideThickness, 0:colN]
                
                res.append(tmp)

        else:

            for i in range(0, colN - self.slideThickness, self.slideThickness):
                # croping is much easier in Python, it is basically just slicing
                tmp = src[0:self.slideThickness, i:i+rowN]
                res.append(tmp)

        return res

    def centerRoadSide(self, src, dire):
        """
        Input:
            - src: np array of images
            - dire: directon constant
        """

        res = []
        inputN = len(src)
        

        for i in range(inputN):
            
            #imgray = cv2.cvtColor(src[i], cv2.COLOR_BGR2GRAY)
            imgray = src[i]
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)

            _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmp = []

            cntsN = len(cnts)
            ## dont know why, I keep it commented
            if (cntsN == 0):
                res.append(tmp)
            #continue
        
            for j in range(cntsN):
                
                area = cv2.contourArea(cnts[j], False)
                
                if (area > 3):
                    
                    M1 = cv2.moments(cnts[j], False)
                    center1 = cv2.KeyPoint(M1["m10"] / M1["m00"], M1["m01"] / M1["m00"], _size=0)
                    (x, y) = center1.pt

                    if (dire == self.VERTICAL):
                        center1 = cv2.KeyPoint(
                            x,
                            y + self.slideThickness * i,
                            _size=0
                        )

                    else:
                        center1 = cv2.KeyPoint(
                            x + self.slideThickness * i,
                            y,
                            _size=0
                        )

                    (x, y) = center1.pt
                    
                    if (x > 0 and y > 0):
                        tmp.append(center1)
            res.append(tmp)

        # np use is important: vector<vector<Point> > will be translated as a np 2D-array of cv2 KeyPoints
        return np.array(res)


    def detectLeftRight(self, points):
        """
        Method used to detect left and right lanes
        Input:
            - points: numpy 2D-array of cv2 KeyPoint instances
        """
        lane1 = []; lane2 = []
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
                if (pointMap[i][j] > max2 and (i != posMax.pt[0] or j != posMax.pt[1]) and postPoint[i][j] == -1): #FIXME "local variable 'posMax' referenced before assignment" possible
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

        line1 = cv2.fitLine(subLane1, 2, 0, 0.01, 0.01)
        line2 = cv2.fitLine(subLane2, 2, 0, 0.01, 0.01)

        try:
            lane1X = (self.BIRDVIEW_WIDTH - line1[3]) * line1[0] / line1[1] + line1[2]
        except:
            lane1X = 0

        try:
            lane2X = (self.BIRDVIEW_WIDTH - line2[3]) * line2[0] / line2[1] + line2[2]
        except:
            lane2X = 0
        
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

    def morphological(self, img):
        dst = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,20)))
        dst = cv2.erode(dst, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 20)))

        return dst

    def transform(self, src_vertices, dst_vertices, src, dst):
        M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
        cv2.warpPerspective(src, dst, M, dst.shape, flags=cv2.INTER_LINEAR, borderValue = cv2.BORDER_CONSTANT)


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


"""
img = cv2.imread('1.png', 1)
detect = DetectLane()
detect.update(img,0)
"""