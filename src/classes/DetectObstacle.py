import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import imutils

class DetectObstacle():

    def __init__(self):
        
        self.height = 240
        self.width = 320
        self.danger = 0
        self.count = 0
        self.keypoint = []
        self.listThreshold = [150,140,130,120,110,100,0]
        self.color = [(255, 255, 0),(0, 191, 255),(0, 128, 255),(0, 64, 255),(0, 0, 255),(153, 51, 204),(0,0,0)]

    ## Apply a threshold of 'value'
    def thresholdImg(self, img, value):

        ret,thresh = cv.threshold(img,value,255,cv.THRESH_BINARY)
        return thresh

    ## Identify danger block
    def filterMask(self, thresh, labels, mask):

        for label in np.unique(labels):



            # For each label, put them in with block on the mask
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # Count the nb of whith pixel
            numPixels = cv.countNonZero(labelMask)
            #print(numPixels)

            # Filter the labels with their size
            if numPixels > 250 and numPixels < 1000:
                ### This labelMask is the danger block of the car !
                mask = cv.add(mask, labelMask)
                self.danger += 1
                self.count = 0

        return mask

    ## Get the contour of the danger block, and add it on the image
    # NOTE: Work only with test image, no result on stream frame
    def contours(self, thresh, mask):

        image, contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        img = cv.cvtColor(mask, cv.COLOR_GRAY2RGB) # Convert image in color
        cv.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv.fillPoly(img, pts =[contours], color=(0, 0, 255))
        return img

    def main(self, img):
        
        self.height = img.shape[0]
        self.width = img.shape[1]

        # Increment the number of frame
        self.count +=1
        #print(self.count)
        #print(self.danger)

        # Cut the border of the frame
        imgCut = img.copy()
        imgCut[0:self.height, 0:self.width/5] = 0 #Left
        imgCut[0:self.height, (4*self.width/5):self.width] = 0 #Right
        imgCut[0:self.height/4, 0:self.width] = 0 #Top
        imgCut[(3*self.height/4):self.height, 0:self.width] = 0 #Bottom

        # cv.imshow("ImageCut", imgCut)

        # Applay the threshold on the cut frame
        thresh = self.thresholdImg(imgCut, self.listThreshold[self.danger])

        # cv.imshow("Th", thresh)
        # Identify block of pixel with the almost same color of grey
        labels = measure.label(thresh, connectivity=2, background=255)
        # Create a black mask
        mask = np.zeros(thresh.shape, dtype="uint8")

        # Add to the mask the danger block
        mask = self.filterMask(thresh, labels, mask)
        #cv.imshow("Cars", mask)


        blobparams = cv.SimpleBlobDetector_Params()
        blobparams.filterByArea = False
        blobparams.filterByInertia = False
        blobparams.filterByCircularity = False
        blobparams.filterByConvexity=False
        blobparams.minDistBetweenBlobs = 500

        detector = cv.SimpleBlobDetector_create(blobparams)
        keypoints = detector.detect(mask)


        if len(keypoints) > 0:
            self.keypoint = keypoints[0]
            # print(self.keypoint.size)
            # print(self.keypoint.pt)
        
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        if self.keypoint != []:

            start_point = (int(self.keypoint.pt[0])-20, int(self.keypoint.pt[1])-20)      
            end_point = (int(self.keypoint.pt[0])+20, int(self.keypoint.pt[1])+20)
            color = self.color[self.danger] 
            thickness = 3

            img = cv.rectangle(img, start_point, end_point, color, thickness)

        # Print 'DANGER' for 50 frame after have seen a danger block
        # NOTE: the value 50 is to change depending of the power of the computer
        if (self.count > 20) and (self.danger > 0):
            self.danger = 0
            self.keypoint = []

        # Print 'DANGER'
        
        if self.danger > 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img,"Danger",(260,20), font, .5,self.color[self.danger],2,cv.LINE_AA)
        # cv.imshow("Depth danger", img)

        return self.keypoint


