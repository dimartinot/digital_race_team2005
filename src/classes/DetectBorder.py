import cv2
import numpy as np

class DetectBorder():
    def __init__(self):
        
        self.blobparams = cv2.SimpleBlobDetector_Params()
        self.blobparams.filterByArea = True
        self.blobparams.minArea = 0
        self.blobparams.filterByInertia = False
        self.blobparams.filterByCircularity = False
        self.blobparams.filterByConvexity=False
        self.blobparams.minDistBetweenBlobs = 300

        self.detector = cv2.SimpleBlobDetector_create(self.blobparams)

        self.slideThickness = 10
        self.BIRDVIEW_WIDTH = 240
        self.BIRDVIEW_HEIGHT = 320
        self.skyLine = 85
        self.VERTICAL = 0
        self.HORIZONTAL = 1
        self.leftLane = []
        self.rightLane = []
    
    def getLeftLane(self):
        return self.leftLane

    def getRightLane(self):
        return self.rightLane

    def delNoise(self, frame):

        for i in range(0,frame.shape[0]):
            for j in range(1,frame.shape[1]-1):
                if frame[i][j]==255:
                    if frame[i][j-1]==frame[i][j+1]==0:
                        frame[i][j] = 0
        return(frame)

    def birdViewTransform(self, src):
        height, width = src.shape

        src_vertices = np.array([
            [0, self.skyLine],
            [width, self.skyLine],
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
        return(warp)

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
    
    def detectBorders(self, points):
    
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

    def threshold(self, frame):

        for i in range(frame.shape[0]):
            maxi = np.amax(frame[i])
            mini = np.amin(frame[i])

            _, output = cv2.threshold(frame[i], maxi-int(7*(maxi-mini)/8), 255, cv2.THRESH_BINARY)

            frame[i] = output.flatten()

            if len(np.where(frame[i]==255)[0])<=5:
                frame[i] = np.array([255]*frame.shape[1])

        return(frame)

    def main(self, frame):
        copy = np.copy(frame)

        heigh = copy.shape[0]

        copy = copy[int(heigh/4):]

        copy = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)

        copy = self.threshold(copy)
        copy = cv2.bitwise_not(copy)

        copy = cv2.resize(copy, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
        
        bird_view = self.birdViewTransform(copy)

        cv2.imshow("Bird View", bird_view)

        self.fillLane(bird_view)

        layers = self.splitLayer(bird_view, self.VERTICAL)

        points = self.centerRoadSide(layers, self.VERTICAL)

        self.detectBorders(points)

        lane = np.zeros(bird_view.shape, dtype=np.uint8)
        lane = cv2.cvtColor(lane, cv2.COLOR_GRAY2BGR)

        # for i in range(points.shape[0]):
        #     for j in range(len(points[i])):
        #         (x,y) = points[i][j].pt
        #         x = int(x)
        #         y = int(y)

        #         cv2.circle(lane, (x,y) , 1, (0,0,255), 2, 8, 0)

        for i in range(1, len(self.leftLane)):
            if (self.leftLane[i] != None):
                cv2.circle(lane, (int(self.leftLane[i][0]), int(self.leftLane[i][1])), 1, (0,0,255), 2, 8, 0)

        for i in range(1, len(self.rightLane)):
                if (self.rightLane[i] != None):
                    cv2.circle(lane, (int(self.rightLane[i][0]), int(self.rightLane[i][1])), 1, (255,0,0), 2, 8, 0)

        # cv2.imshow("Lane Detect", lane)

        return(lane)

if __name__ == "__main__":

    img = cv2.imread('./data/test_border.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detect = DetectBorder()

    while True:



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cv2.destroyAllWindows()