import cv2
import numpy as np

class DetectIntersection():
    def __init__(self):
        
        self.memory = np.zeros(3)
        self.sign_detected = False
        self.count=0

        self.blobparams = cv2.SimpleBlobDetector_Params()
        self.blobparams.filterByArea = True
        self.blobparams.minArea = 100
        self.blobparams.filterByCircularity = False
        self.blobparams.filterByInertia=True
        self.blobparams.minInertiaRatio = 0.7
        self.blobparams.maxInertiaRatio = 1
        self.blobparams.filterByConvexity=True
        self.blobparams.minDistBetweenBlobs = 500

        self.detector = cv2.SimpleBlobDetector_create(self.blobparams)

        self.params= {
            'lB': 50,
            'lG': 50,
            'lR': 50,
            'hB': 127,
            'hG': 127,
            'hR': 127
        }

        self.confirmation = False
    

    def checking(self, depth):
        """ TODO: comments
        """
        # copy the input data
        detect = np.copy(depth)

        # cropping
        heigh = detect.shape[0]
        width = detect.shape[1]
        detect = detect[:int(heigh/2), int(width/2):]

        # delete too far objects
        detect[detect > 150] = 0

        # threshold
        lowerLimits = np.array([self.params['lB'], self.params['lG'], self.params['lR']])
        upperLimits = np.array([self.params['hB'], self.params['hG'], self.params['hR']])
        thresholded = cv2.inRange(detect, lowerLimits, upperLimits)

        thresholded = cv2.bitwise_not(thresholded)

        cv2.imshow('TEST', thresholded)

        # find circles
        circles = self.detector.detect(thresholded)

        if len(circles) > 0:
            return(True)
        
        return(False)

    def main(self, frame, sign_detection):
        output = None

        if sign_detection:
            self.memory[0] = self.memory[1]
            self.memory[1] = self.memory[2]
            self.memory[2] = -1 if sign_detection=='left' else 1
            self.sign_detected = True
            self.count = 0
        
        
        if self.sign_detected and not sign_detection:
            
            if self.count >= 6: 
                output = 'right' if self.memory.mean() > 0 else 'left'
                self.sign_detected = False
                self.count = 0
            else:
                self.count += 1
        
        # cv2.putText(frame, output, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return(output)
        