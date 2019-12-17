from datetime import datetime
import numpy as np


class StampImageCollection:
    def __init__(self, images):
        self.images = images
        self.timeStamp = datetime.now()
        self.mergedImage = None

    def set_images(self, image):
        self.images = image

    def get_images(self):
        return self.images

    def merge_image(self):
        cam_left = self.images[0]
        cam_mid = self.images[1]
        cam_right = self.images[2]
        x = 0
        y = 0
        cam_left = cam_left.get_image()[y:y + cam_left.height, x:x + cam_left.width]
        cam_mid = cam_mid.get_image()[y:y + cam_mid.height, x:x + cam_mid.width]
        cam_right = cam_right.get_image()[y:y + cam_right.height, x:x + cam_right.width]
        img = np.concatenate((cam_left, cam_mid, cam_right), axis=1)
        np.asarray(img)

        self.mergedImage = img

    def get_merged_image(self):
        if self.mergedImage is None:
            raise FileNotFoundError
        else:
            return self.mergedImage


class StampImage:
    def __init__(self, frame, camera_id, x_offset, y_offset, width, height):
        self.images = frame
        self.cameraId = camera_id
        self.xOffset = x_offset
        self.yOffset = y_offset
        self.width = width
        self.height = height

    def set_image(self, image):
        self.images = image

    def set_camera_id(self, camera):
        self.cameraId = camera

    def set_x_offset(self, x):
        self.xOffset = x

    def set_y_offset(self, y):
        self.yOffset = y

    def get_image(self):
        return self.images
