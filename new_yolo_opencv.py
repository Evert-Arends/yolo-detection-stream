from queue import Queue
from time import sleep
import threading
import time

import cv2
import argparse
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

frameGrabsQueue = Queue()
frameToShowQueue = Queue()

exitFlag = 0
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()


# wget https://pjreddie.com/media/files/yolov3.weights

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, inner_cv):
    label = str(classes[class_id])
    label = label + " {0}".format(confidence)

    color = COLORS[class_id]

    inner_cv.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    inner_cv.putText(img, label, (x - 10, y - 10), inner_cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


class FrameGrabThread(threading.Thread):
    def __init__(self, thread_id, name, video_stream, inner_cv2, inner_scale, grab_count_per_second=30):
        threading.Thread.__init__(self)
        self.scale = inner_scale
        self.cv2 = inner_cv2
        self.vs = video_stream
        self.grab_count_per_second = grab_count_per_second
        self.sleepTime = 60 / self.grab_count_per_second
        self.threadID = thread_id
        self.name = name

    def run(self):
        print("Starting " + self.name)
        while True:
            self.grab_frame()

    def grab_frame(self):
        print("Should grab a screen")
        _image = frame = self.vs.read()
        frameGrabsQueue.put(_image)
        sleep(.450)


class FrameScanThread(threading.Thread):
    def __init__(self, thread_id, name, net, cv2, scale, colors):
        threading.Thread.__init__(self)
        self.scale = scale
        self.cv2 = cv2
        self._colors = colors
        self._net = net
        self.threadID = thread_id
        self.name = name

    def run(self):
        while True:
            self.analyze_frame()

    def analyze_frame(self):
        if not frameGrabsQueue.empty():
            print("The framegrab current queue: {0}".format(frameGrabsQueue.qsize()))
            print("analyzing frame grab....")
            _image = frameGrabsQueue.get()
            _blob = self.cv2.dnn.blobFromImage(_image, self.scale, (288, 288), (0, 0, 0), True, crop=False)
            self._net.setInput(_blob)
            outs = self._net.forward(get_output_layers(self._net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = self.cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            back_img = _image
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(back_img, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                                round(y + h),
                                self.cv2)

            frameToShowQueue.put(back_img)


# cv2.dnn.DNN_TARGET_OPENCL = True


class FrameShowThread(threading.Thread):
    def __init__(self, thread_id, name, cv2):
        threading.Thread.__init__(self)
        self.cv2 = cv2
        self.threadID = thread_id
        self.name = name

    def run(self):
        while True:
            self.grab_frame()

    def grab_frame(self):
        if frameToShowQueue.empty():
            print("Nothing to show, queue empty")
            sleep(0.4)
        else:
            i_image = frameToShowQueue.get()
            self.cv2.imwrite("object-detection.jpg", i_image)
            self.cv2.imshow("object detection", i_image)
            self.cv2.waitKey() & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                print("yo")


if __name__ == '__main__':
    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net1 = cv2.dnn.readNet(args.weights, args.config)
    net2 = cv2.dnn.readNet(args.weights, args.config)
    net3 = cv2.dnn.readNet(args.weights, args.config)
    net1.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    net3.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    net4 = cv2.dnn.readNet(args.weights, args.config)
    net5 = cv2.dnn.readNet(args.weights, args.config)
    net6 = cv2.dnn.readNet(args.weights, args.config)
    net4.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    net5.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    net6.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    fps = FPS().start()

    # vs = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()
    vs = VideoStream(src=0).start()
    sleep(2)
    first_frame = vs.read()
    Width = first_frame.shape[1]
    Height = first_frame.shape[0]
    scale = 0.00392
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # frameShowThread = FrameShowThread(3, "FrameShowThread", cv2)
    # frameShowThread.setDaemon(True)
    # frameShowThread.start()

    frameGrabThread = FrameGrabThread(0, "FrameGrabThread", vs, cv2, scale, 30)
    frameGrabThread.setDaemon(True)
    frameGrabThread.start()
    frameScanThread = FrameScanThread(1, "scan", net1, cv2, scale, COLORS)
    frameScanThread.setDaemon(True)
    frameScanThread.start()
    frameScanThread2 = FrameScanThread(1, "scan", net2, cv2, scale, COLORS)
    frameScanThread2.setDaemon(True)
    frameScanThread2.start()
    frameScanThread3 = FrameScanThread(1, "scan", net3, cv2, scale, COLORS)
    frameScanThread3.setDaemon(True)
    frameScanThread3.start()
    frameScanThread4 = FrameScanThread(1, "scan", net4, cv2, scale, COLORS)
    frameScanThread4.setDaemon(True)
    frameScanThread4.start()
    frameScanThread5 = FrameScanThread(1, "scan", net5, cv2, scale, COLORS)
    frameScanThread5.setDaemon(True)
    frameScanThread5.start()
    frameScanThread6 = FrameScanThread(1, "scan", net6, cv2, scale, COLORS)
    frameScanThread6.setDaemon(True)
    frameScanThread6.start()

    # sleep(1)
    frameToShowQueue.put(first_frame)

    while True:
        if frameToShowQueue.empty():
            print("Nothing to show, queue empty")
        else:
            i_image = frameToShowQueue.get()
            cv2.imwrite("object-detection.jpg", i_image)
            cv2.imshow("object detection", i_image)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
