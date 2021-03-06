# python yolo_last_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
import _thread
from queue import Queue
from time import sleep
import threading
import time
import datetime

import requests

import StampImageCollection as StampImageLibrary

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
    print("...")
    print("X and Y Coordinates: X: {0}, Y: {1}".format(x, y))
    print("...")

    label = str(classes[class_id])
    label = label + " {0}".format(confidence)

    color = COLORS[class_id]
    # x = (x + img.width)
    inner_cv.rectangle(img.get_image(), (x, y), (x_plus_w, y_plus_h), color, 2)

    inner_cv.putText(img.get_image(), label, (x - 10, y - 10), inner_cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate center point pixel x, y
    x1 = x
    x2 = x_plus_w
    y1 = y
    y2 = y_plus_h

    xCenter = int((x1 + x2) / 2)  # Center X coord of detection box.
    yCenter = int((y1 + y2) / 2)  # Center Y coord of detection box.
    # See if correct X,Y by drawing a circle.
    inner_cv.circle(img.get_image(), (xCenter, yCenter), 5, color, -1)

    def post_the_thing(value):
        url = "http://dwnb.nl:8880/measurements/"
        sensor_id = 'K4rXHCiEylki6ewmPbCxAR7HYKhjRHoMsNUXkqaGzbR7vYEFPvf0Y1zF'
        secret = '9cNTPuyfWRHxrhNOymBR2s7UjnpCyywA61BmXkQR3YZglP'

        payload = "unitTypes_UnitShort=locatie&measureingType_Type=locatie&value="+str(value)+"&sensors_id="+sensor_id+"&location=datalab&secret="+secret
        headers = {
            'Content-Type': "application/x-www-form-urlencoded",
            'Accept': "*/*"
        }

        response = requests.request("POST", url, data=payload, headers=headers)
        print(response) #200 = success

    # Add coords to "JSON string"
    json_value = {'x': xCenter, 'y': yCenter, 'datetime': img.get_datetime()} # TODO: Add dateTime

    # JSON post
    _thread.start_new_thread(post_the_thing, (json_value,))

    return img


class FrameGrabThread(threading.Thread):
    def __init__(self, thread_id, name, video_streams, inner_cv2, inner_scale, grab_count_per_second=30):
        threading.Thread.__init__(self)
        self.scale = inner_scale
        self.cv2 = inner_cv2
        self.vs = video_streams
        self.grab_count_per_second = grab_count_per_second
        self.sleepTime = 60 / self.grab_count_per_second
        self.threadID = thread_id
        self.name = name
        self.should_run = True
        self.should_run1 = True

    def run(self):
        print("Starting " + self.name)
        while True:
            self.grab_frame()

    def grab_frame(self):
        print("Should grab a screen")
        datetime_now = str(datetime.datetime.now())
        if self.should_run:
            cam_left = StampImageLibrary.StampImage(
                frame=self.vs[0].read(),
                camera_id=0,
                x_offset=0,
                y_offset=0,
                width=Width,
                height=Height,
                current_datetime=datetime_now
            )
            cam_mid = StampImageLibrary.StampImage(
                frame=self.vs[0].read(),
                camera_id=1,
                x_offset=800,
                y_offset=0,
                width=Width,
                height=Height,
                current_datetime=datetime_now
            )
            cam_right = StampImageLibrary.StampImage(
                frame=self.vs[0].read(),
                camera_id=2,
                x_offset=1600,
                y_offset=0,
                width=Width,
                height=Height,
                current_datetime=datetime_now
            )

            frameGrabsQueue.put(cam_right)
            frameGrabsQueue.put(cam_mid)
            frameGrabsQueue.put(cam_left)
            self.should_run = False
        else:
            if not self.should_run:
                self.should_run = True

        sleep(.40)


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
            print("The frame grab current queue: {0}".format(frameGrabsQueue.qsize()))
            print("analyzing frame grab....")

            _image = frameGrabsQueue.get()

            _blob = self.cv2.dnn.blobFromImage(_image.get_image(), self.scale, (288, 288), (0, 0, 0), True, crop=False)
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


if __name__ == '__main__':
    classes = None

    net_count = 5
    thread_count = 5
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    print("Initializing networks....")

    net_dict = []
    for item in range(net_count):
        net_dict.append(cv2.dnn.readNet(args.weights, args.config))
        net_dict[item].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Done..")

    fps = FPS().start()

    # vs = VideoStream(src="http://208.105.238.178:9080/mjpg/video.mjpg").start()
    # vs = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()
    # vs1 = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()
    # vs2 = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()
    vs = VideoStream(src=0).start()
    vs1 = vs
    # vs2 = VideoStream(src=0).start()
    sleep(2)
    first_frame = vs.read()
    Width = first_frame.shape[1]
    Height = first_frame.shape[0]
    scale = 0.00392
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # frameShowThread = FrameShowThread(3, "FrameShowThread", cv2)
    # frameShowThread.setDaemon(True)
    # frameShowThread.start()

    print("Starting frame-scan threads: ({0})".format(thread_count))
    for thread in range(thread_count):
        sleep(0.1)
        frameScanThread7 = FrameScanThread(1, "scan", net_dict[thread], cv2, scale, COLORS)
        frameScanThread7.setDaemon(True)
        frameScanThread7.start()

    sleep(5)

    print("Starting frame grab thread")
    # vs_list = [vs, vs1, vs2]
    frameGrabThread = FrameGrabThread(0, "FrameGrabThread", [vs, vs1], cv2, scale, 30)
    frameGrabThread.setDaemon(True)
    frameGrabThread.start()
    print("Done..")

    # frameToShowQueue.put(first_frame)
    print("Running show now!")
    while True:
        if not frameToShowQueue.empty():
            i_image = frameToShowQueue.get()
            cv2.imwrite("object-detection.jpg", i_image.get_image())
            cv2.imshow("object detection", i_image.get_image())
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
