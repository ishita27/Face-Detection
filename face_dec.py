from __future__ import division
import cv2
import time
import sys

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    # blobFromImages (InputArrayOfArrays images, double scalefactor=1.0, Size size=Size(),
    # const Scalar &mean=Scalar(), bool swapRB=false, bool crop=false, int ddepth=CV_32F)
    # creates 4-D blobs from images,sometimes crops from centre

    net.setInput(blob)  #Sets the new input value for the network.
    detections = net.forward()  #Runs forward pass to compute output of layer with name outputName.
    bboxes = [] #bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #the output blob was 4-D, i is the number of faces and 4th gives bbox info and score
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

if __name__ == "__main__" :

    DNN = "CAFFE"
    if DNN == "CAFFE":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    # else:
    #     modelFile = "models/opencv_face_detector_uint8.pb"
    #     configFile = "models/opencv_face_detector.pbtxt"
    #     net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture('rowing.mp4')
    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

    frame_count = 0
    tt_opencvDnn = 0
    while(cap.isOpened()):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Face Detection", outOpencvDnn)

        vid_writer.write(outOpencvDnn)
        if frame_count == 1:
            tt_opencvDnn = 0

            k = cv2.waitKey(10)
        if k == 30:
            break
    cv2.destroyAllWindows()
    vid_writer.release()
