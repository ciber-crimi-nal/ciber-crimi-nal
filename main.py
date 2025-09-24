import os
import cv2
from scripts.dni_detector import read_dni

sides = ["front","back"]
ruta = ["images/sintetico.png", "images/sintetico_rever.png"]


print("************ CENSURAR yolo + homografia + opencv ************")
dni_img_censurado = read_dni(ruta, sides, debug=False)
if dni_img_censurado is not None and isinstance(dni_img_censurado, dict):
    
    name = os.path.basename(ruta[0]).split(".")[0]

    for side, img in dni_img_censurado.items():
        if img is not None:
            print(side)

            output = os.path.join("output", f"{name}_{side}_censurado.png")
            cv2.imwrite(output, img)
else:
    print("La imagen censurada está vacía.")