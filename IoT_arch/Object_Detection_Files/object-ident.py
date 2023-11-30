import cv2
from gpiozero import LED

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/quantumai/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/quantumai/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/quantumai/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(200,200)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def capInit():

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    return cap
    
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo
led_pin = 17
led = LED(led_pin)

if __name__ == "__main__":

    cap = capInit()


    while True:
        success, img = cap.read()
        if not success:
            cap.release()
            cap.open(0)
            print("Error: Read failed")
           #cap = capInit()
            continue
        result, objectInfo = getObjects(img,0.45,0.2,objects=['person'])
        #'car', 'motorcycle','truck','bus'
        print(len(objectInfo))
        # Check if 'person' is in the detected objects
        if any(obj[1] == 'person' for obj in objectInfo):
            #Turn on the LED
            led.on()
        else:
            #Turn off the LED
            led.off()
            
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)