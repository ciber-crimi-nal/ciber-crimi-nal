import cv2
from datetime import *
import matplotlib.pyplot as plt

def print_debug(debug: bool, message: str):
    if debug:
        print("#DEBUG", message)


def show_debug(debug: bool, image):
    if debug:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

def positive_coords(list_points: list[int]) -> list[int]:
    new_points = []

    for point in list_points:
        if point < 0:
            new_points.append(0)
        else:
            new_points.append(point)

    return new_points