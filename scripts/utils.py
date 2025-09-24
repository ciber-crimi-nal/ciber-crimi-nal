from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
import os
import math
from PIL import Image
from io import BytesIO
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

def pdf_generator(images, name, text, path="output"):
    pdf_path = os.path.join(path, f"{name}.pdf")
    os.makedirs(path, exist_ok=True)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    a4_width, a4_height = A4
    scale_factor = 0.45
    spacing = 20

    y_cursor = a4_height 

    for image in images:
        img_rgb = bgr_to_rgb(image)
        pil_img = Image.fromarray(img_rgb)

        img_buffer = BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_reader = ImageReader(img_buffer)

        img_w, img_h = pil_img.size
        img_ratio = img_w / img_h

        target_width = a4_width * scale_factor
        target_height = target_width / img_ratio

        x = (a4_width - target_width) / 2
        y_cursor -= target_height + spacing

        if y_cursor < 0:
            c.showPage()
            y_cursor = a4_height - target_height - spacing

        c.drawImage(img_reader, x, y_cursor, width=target_width, height=target_height)

    max_font_size = 45
    min_font_size = 20
    font_name = "Helvetica"
    font_size = max_font_size
    text_width = stringWidth(text, font_name, font_size)

    while text_width > a4_width * 0.4 and font_size > min_font_size:
        font_size -= 2
        text_width = stringWidth(text, font_name, font_size)

    angle = 45  # grados
    rad = math.radians(angle)

    text_height = font_size

    rotated_width = abs(text_width * math.cos(rad)) + abs(text_height * math.sin(rad))
    rotated_height = abs(text_width * math.sin(rad)) + abs(text_height * math.cos(rad))

    margin = 10
    x_translate = margin + rotated_width / 2
    y_translate = a4_height - margin - rotated_height / 2

    c.saveState()
    c.translate(x_translate, y_translate)
    c.rotate(angle)

    c.setFont(font_name, font_size)
    c.setFillColorRGB(0.6, 0.6, 0.6, alpha=0.5)
    c.drawCentredString(0, 0, text) 
    c.restoreState()

    c.save()
    print(f"PDF generado: {pdf_path}")