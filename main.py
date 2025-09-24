import cv2
from scripts.dni_detector import read_dni

sides = ["front"]

print("************ CENSURAR yolo + homografia + opencv ************")
ruta = ["images/real.jpeg"]
# ruta = ["images/sintetico.png"]
dni_img_censurado = read_dni(ruta, sides, debug=False)
if dni_img_censurado is not None:
    cv2.imwrite("output/dni_censurado2.png", dni_img_censurado)
else:
    print("La imagen censurada está vacía.")