from PIL import Image
import os
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

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def pil_converter(image):
        return Image.fromarray(image)

def pdf_generator(images,name,text,path="output"):
    pdf_images = []

    for image in images:
        img_rgb = bgr_to_rgb(image)
        pil_img = pil_converter(img_rgb)
        pdf_images.append(pil_img)

    if pdf_images:
        pdf_path = os.path.join(f"{path}/{name}.pdf")
        pdf_images[0].save(pdf_path, save_all=True, append_images=pdf_images[1:])
        print(f"PDF generado: {pdf_path}")
    else:
        print("No se generó el PDF porque no hay imágenes válidas.")
