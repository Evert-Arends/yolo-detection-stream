from queue import Queue
from time import sleep
import threading
import time
import datetime
import StampImageCollection as StampImageLibrary
import functools

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

    return img

# grab frames from cams, build an object with the specified data.
class FrameGrabThread(threading.Thread):
    def __init__(self, thread_id, name, video_streams, inner_cv2, inner_scale, grab_count_per_second=30):
        threading.Thread.__init__(self)
        self.scale = inner_scale
        self.cv2 = inner_cv2
        self.vs = video_streams
        self.grab_count_per_second = grab_count_per_second
        self.sleepTime = 60 / self.grab_count_per_second # ignore until this comment is removed.
        self.threadID = thread_id
        self.name = name
        self.should_run = True
        self.should_run1 = True

    def run(self):
        print("Starting " + self.name)
        while True:
            self.grab_frame()

    # grab frame from cam, and build a frame object
    def grab_frame(self):
        sleep(.42)
        print("Should grab a screen")
        # no one can explain this to me, but without the booleans the threads appear to be blocking, do yourself a favor
        # and DON'T TOUCH IT
        if self.should_run and self.should_run1:
            cam_left = StampImageLibrary.StampImage(
                frame=self.vs[0].read(),
                camera_id=0,
                x_offset=0,
                y_offset=0,
                width=Width,
                height=Height,
                current_datetime=datetime.datetime.now()
            )
            cam_mid = StampImageLibrary.StampImage(
                frame=self.vs[1].read(),
                camera_id=1,
                x_offset=800,
                y_offset=0,
                width=Width,
                height=Height,
                current_datetime=datetime.datetime.now())
            cam_right = StampImageLibrary.StampImage(
                frame=self.vs[2].read(),
                camera_id=2,
                x_offset=1600,
                y_offset=0,
                width=Width,
                height=Height,
                current_datetime=datetime.datetime.now()
            )

            images_collection = StampImageLibrary.StampImageCollection([cam_left, cam_mid, cam_right])
            for item in images_collection.images:
                frameGrabsQueue.put(item)

            # Skip every 2nd image.
            self.should_run = False
        else:
            if not self.should_run:
                self.should_run = True



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

    # this method runs in a separate thread, analyzes frames and saves them into a queue.
    def analyze_frame(self):
        if not frameGrabsQueue.empty():
            print("The frame grab current queue: {0}".format(frameGrabsQueue.qsize()))
            print("analyzing frame grab....")

            # retrieve frame from Queue
            _image = frameGrabsQueue.get()

            _blob = self.cv2.dnn.blobFromImage(_image.get_image(), self.scale, (288, 288), (0, 0, 0), True, crop=False)
            self._net.setInput(_blob)
            outs = self._net.forward(get_output_layers(self._net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            # run through detections
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

                        # get center of detected person
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


# Enable for CUDA
# cv2.dnn.DNN_TARGET_OPENCL = True

def setup_nets(weights, config):
    return cv2.dnn.readNet(weights, config)


if __name__ == '__main__':
    classes = None

    net_count = 15
    thread_count = 15

    # Read labels from textfile
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    print("Initializing networks.... {0}".format(net_count))

    # Load darknets into ram, specified in the net_count
    net_dict = []
    for item in range(net_count):
        net_dict.append(setup_nets(args.weights, args.config))
        net_dict[item].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Done..")

    fps = FPS().start()

    vs1 = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()
    vs = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()
    vs2 = VideoStream(src="http://74.92.195.57:81/mjpg/video.mjpg").start()

    # Give it time to retrieve frames from stream
    sleep(1)

    first_frame = vs.read()
    Width = first_frame.shape[1]
    Height = first_frame.shape[0]
    scale = 0.00392
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    print("Starting frame-scan threads: ({0})".format(thread_count))

    # Spawning the (n) amount of threads to read from the Queue, and scan for detections
    for thread in range(thread_count):
        sleep(0.01)
        frameScanThread7 = FrameScanThread(1, "scan", net_dict[thread], cv2, scale, COLORS)
        frameScanThread7.setDaemon(True)
        frameScanThread7.start()

    sleep(1)

    print("Starting frame grab thread")

    # Starting the frameGrabThread, which reads frames from the stream
    frameGrabThread = FrameGrabThread(0, "FrameGrabThread", [vs2, vs1, vs], cv2, scale, 30)
    frameGrabThread.setDaemon(True)
    frameGrabThread.start()
    print("Done..")

    # Retrieve frame to show from Queue, whenever possible for the fastest video feeling
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
