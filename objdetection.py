'''
Name:-Raj Ghadi
Project Title: - Object Detection Using OpenCV
Internship Details: -
                    Company Name= The Sparks Foundation
                    Program Name= Graduate Rotational Internship Program
                    Internship Name= IoT and Computer Vision
                    Task 1= Object Detction
                    Task Descripation= Implement an object detector which identifies the classes of the objects in
                                       an image or video.
'''

#import all the required libraries
import cv2
#import open cv library
import numpy as np
#import numpy library
weights_path = 'yolov3-spp (1).weights'
#initialise weights
configuration_path = 'yolov3.cfg.txt'
#initialise bias
net = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
#Building the DarkNet network with OpenCV according to YOLO paper.
classes = []
with open('coco.names.txt','r') as f:
    classes  = f.read().splitlines()
#creating list of all labels / classes of objects like car,bike,mobile etc

#here in the loop we can pass single frame of image or multiple-continuous frames of image i.e, video
cap=cv2.VideoCapture("test4.mp4")
#here we have option to pass video or to use webcam as input
#img=cv2.imread('1.png')
#but now here my aim is to detect objects from video so i use while loop to get multiple-continuous frames of image i.e, video
while True:
     _, img = cap.read()#
     height, width, _ = img.shape
     #scale range of image/Taking the height and width of the input image
     blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0,0,0), swapRB=True, crop=False)
     # Creating a BLOB(Binary large object) by making neccesary transforms and scaling like normalization (divide by 255),
     # and scaling size to (416, 416) and converting BGR image to RGB
     net.setInput(blob)
     #add all changes/neccesary transforms and scaling in net
     output_layers_names=net.getUnconnectedOutLayersNames()
     # Get the index of the output layers and  the name of all layers of the network.
     layeroutputs=net.forward(output_layers_names)
     #Running the pretrainned model with our input

     boxes=[]
     confidences=[]
     class_ids=[]

     for output in layeroutputs:# selecting the output layer out of 3
        for detection in output:# selecting a grid
           scores =detection[5:]#Selecting 80 classes
           class_id=np.argmax(scores)#Finding maximum probability position
           confidence=scores[class_id]#maximum probability
           if confidence >0.5:#Class thresold check
              center_x =int(detection[0]*width)
              center_y=int(detection[1]*height)
              w=int(detection[2]*width)
              h=int(detection[3]*height)

              x=int(center_x -w/2)
              y=int(center_y -h/2)

              boxes.append([x,y,w,h])
              confidences.append((float(confidence)))#add to list confidence
              class_ids.append(class_id)#add to list class_id


     indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
     font=cv2.FONT_HERSHEY_PLAIN #set font
     colors=np.random.uniform(0,255, size=(len(boxes),3))#give random color to rectangle

     for i in indexes.flatten():
         x,y,w,h = boxes[i]
         label = str(classes[class_ids[i]])#store name of detected object
         confidence=str(round(confidences[i],2))#store confidence of detected object
         color=colors[i]#give color to rectangle of detected object
         cv2.rectangle(img,(x,y),(x+w, y+h),color,2)#create rectangle to detected object with proper scaling
         cv2.putText(img, label + "" + confidence, (x, y+20), font, 2, (255,255,255), 2)#show all provided and store information of detected object

     cv2.imshow('Image',img)#show image
     key = cv2.waitKey(1)
     if key == 27:
         break
#if ESC key is pressed then video will stop and all windows of output will stop running
cap.release()
cv2.destroyAllWindows()

# Thank You!!!