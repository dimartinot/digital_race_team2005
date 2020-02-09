import cv2
import numpy as np

class DetectSign():
    def __init__(self):

        self.blobparams = cv2.SimpleBlobDetector_Params()
        self.blobparams.filterByArea = True
        self.blobparams.minArea = 100
        self.blobparams.maxArea = 400

        #### Default ####
        self.blobparams.filterByInertia = False
        self.blobparams.filterByCircularity = False
        #################

        ###### New ######
        # self.blobparams.filterByCircularity = True
        # self.blobparams.minCircularity = 0.4
        # self.blobparams.maxCircularity = 1

        # self.blobparams.filterByInertia=True
        # self.blobparams.minInertiaRatio=0.4
        # self.blobparams.maxInertiaRatio=1
        #################

        self.blobparams.filterByConvexity=False
        self.blobparams.minDistBetweenBlobs = 500

        self.detector = cv2.SimpleBlobDetector_create(self.blobparams)
        
        self.params= {
            'lB': 0,
            'lG': 36,
            'lR': 0,
            'hB': 42,
            'hG': 255,
            'hR': 128,
            'shadow': 0
        }

        self.spikes = {
            'direction': None,
            'count': 0
        }

        self.memory = np.zeros(5)
    
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


    def remBackground(self, arr):
        shape = arr.shape
        new_arr = np.zeros(shape)

        for i in range(shape[0]):
            if 0 in arr[i]:

                for j in range(shape[1]):
                    if arr[i][j] == 0:
                        first = j
                        break
                for j in range(shape[1]-1, -1, -1):
                    if arr[i][j] == 0:
                        last = j
                        break
                
                for j in range(first, last+1):
                    new_arr[i][j] = arr[i][j]

        return(new_arr)

    def recognize(self, thresholded):
        """ TODO: comments
        """
        keypoints = self.detector.detect(thresholded)

        if len(keypoints)>0:
            x = int(keypoints[0].pt[0])
            y = int(keypoints[0].pt[1])
            margin = int(keypoints[0].size / 2) + 5
            # crop image to get the sign
            sign = thresholded[y-margin:y+margin, x-margin:x+margin]
            
            # cv2.imshow('Sign', sign)
            # take lower part of the sign
            sign = sign[margin:]

            sign = self.remBackground(sign)

            white_pixels = np.where(sign==255)

            decision = None

            if len(white_pixels[1])>0:

                g_center = np.sum(white_pixels[1])/len(white_pixels[1])

                decision = 'left' if g_center>=sign.shape[1]/2 else 'right'
                

            return(decision)

                # print('Decision:', decision)
                # cv2.putText(original,decision, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # detect = cv2.drawKeypoints(detect, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # cv2.imshow('Original', frame)
        # cv2.imshow('Mask', thresholded)
        # cv2.imshow('Cropped', detect)
        # cv2.imshow('Detect sign', original)

    def fire(self, frame, thresholded):
        """ TODO: comments
        """
        decision = self.recognize(thresholded)
        
        # cv2.putText(frame,decision, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        self.memory[0] = self.memory[1]
        self.memory[1] = self.memory[2]
        self.memory[2] = self.memory[3]
        self.memory[3] = self.memory[4]

        if decision:
            """ 
            if decision == self.spikes['direction']:
                self.spikes['count']+=1
            else:
                self.spikes['direction']=decision
                self.spikes['count']=1
            """
            self.memory[4] = -1 if decision=='left' else 1
        else:
            # self.spikes['direction']=decision
            # self.spikes['count']=0
            self.memory[4] = 0

        """
        fire = False
        if self.spikes['count']>=3 :
            fire = True
            self.spikes['count']=0

        return(self.spikes['direction'] if fire else None)
        """

        mean = self.memory.mean()

        if abs(mean)<0.3:
            return(None)
        else:
            if mean>0:
                return('right')
            else:
                return('left')

    def main(self, frame):
        """ TODO: comments
        """
        detect = np.copy(frame)

        heigh = frame.shape[0]
        width = frame.shape[1]

        detect = detect[:int(heigh/2), int(width/2):]

        frame_hsv = cv2.cvtColor(detect, cv2.COLOR_RGB2HSV)

        lowerLimits = np.array([self.params['lB'], self.params['lG'], self.params['lR']])
        upperLimits = np.array([self.params['hB'], self.params['hG'], self.params['hR']])

        # Our operations on the frame come here
        thresholded = cv2.inRange(frame_hsv, lowerLimits, upperLimits)

        thresholded = cv2.bitwise_not(thresholded)

        output = self.fire(frame, thresholded)

        # cv2.putText(frame,output, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        return(output)
        
        


if __name__ == "__main__":

    sd = DetectSign()


    while True:
        frame = cv2.imread('./img/simu15.png')
        sd.main(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cv2.destroyAllWindows()