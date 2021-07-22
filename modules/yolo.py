import numpy as np
import cv2 as cv
import os


class YoloCV:

    def __init__(self, path, conf=0.5, thresh=0.3):
        self.confidence=conf
        self.threshold=thresh

        self.labels=open(os.path.sep.join([path, "coco.names"])).read().split("\n")

        np.random.seed(76)
        self.colors=np.random.randint(0,255,size=(len(self.labels),3),dtype="uint8")

        self.net=cv.dnn.readNetFromDarknet(os.path.sep.join([path,"yolov3.cfg"]),os.path.sep.join([path,"yolov3.weights"]))

    def detect(self, img):
        (H,W)=img.shape[:2]

        names=self.net.getLayerNames()
        names=[names[i[0] - 1]for i in self.net.getUnconnectedOutLayers()]

        blob_image=cv.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)

        self.net.setInput(blob_image)
        outputs=self.net.forward(names)

        (detections,boxs,class_idxs,confs)=[],[],[],[]

        for output in outputs:
            for detection in output:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]

                if confidence>self.confidence:

                    (centerX,centerY,width,height)=(detection[0:4]*np.array([W,H,W,H])).astype("int")

                    x=int(centerX-(width/2));y=int(centerY-(height/2))

                    boxs.append([x,y,int(width),int(height)])
                    confs.append(float(confidence))
                    class_idxs.append(class_id)

        ids=cv.dnn.NMSBoxes(boxs,confs,self.confidence,self.threshold)

        if len(ids)>0:
            for i in ids.flatten():

                rect_start=(boxs[i][0],boxs[i][1])
                rect_end=(rect_start[0]+boxs[i][2],rect_start[1]+boxs[i][3])

                class_=self.labels[int(class_idxs[i])]

                detections.append(([rect_start,rect_end],class_))

        return detections
