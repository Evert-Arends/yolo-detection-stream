from datetime import datetime


class Image:
    def __init__(self):
        self.image = None
        self.timeStamp = datetime.now()

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image
