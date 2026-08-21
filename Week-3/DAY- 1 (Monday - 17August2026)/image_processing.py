import numpy as np
from PIL import Image


class ImageProcessor:

    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.array = np.array(self.image)

    def grayscale(self):

        gray = np.mean(self.array, axis=2).astype(np.uint8)

        return Image.fromarray(gray)

    def negative(self):

        negative = 255 - self.array

        return Image.fromarray(negative)

    def flip_horizontal(self):

        flipped = np.fliplr(self.array)

        return Image.fromarray(flipped)

    def flip_vertical(self):

        flipped = np.flipud(self.array)

        return Image.fromarray(flipped)

    def rotate90(self):

        rotated = np.rot90(self.array)

        return Image.fromarray(rotated)