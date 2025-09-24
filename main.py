import os
import cv2
from scripts.dni_detector import read_dni
from scripts.utils import pdf_generator

sides = ["front","back"]
ruta = ["images/sintetico.png", "images/sintetico_rever.png"]
images_for_pdf = []


print("************ CENSURAR yolo + homografia + opencv ************")
dni_img_censurado = read_dni(ruta, sides, debug=False)
if dni_img_censurado is not None and isinstance(dni_img_censurado, dict):
    
    name = os.path.basename(ruta[0]).split(".")[0]

    for side, img in dni_img_censurado.items():
        if img is not None:
            output = os.path.join("output", f"{name}_{side}_censurado.png")
            cv2.imwrite(output, img)
            images_for_pdf.append(img)

    if images_for_pdf:
        pdf_generator(images_for_pdf, name, "marca de agua")

else:
    print("La imagen censurada está vacía.")