"""
Synthetic Data
=================
Este módulo se encarga de:
- Generar los datos textuales aleatorios (nombres, fechas, direcciones).
- Calcular códigos de seguridad (MRZ, DNI, CAN).
- Procesar gráficamente la fotografía de la persona.
- Componer todos los elementos sobre la plantilla visual final.
"""

__author__ = "Fernando Broncano"
__copyright__ = "Copyright 2025, Universidad de Extremadura"
__credits__ = ["Antonio Calvo", "Fernando Broncano", "Sergio Guijarro"]
__version__ = "1.0.1"
__maintainer__ = "Antonio Calvo"
__email__ = "acalvopic@unex.es"
__status__ = "Development"

import datetime
import math
import os
import random
import string
from random import randrange, choices
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from dateutil.relativedelta import relativedelta
from keras.utils import img_to_array
from matplotlib import pyplot as plt

from mrz.generator.td1 import TD1CodeGenerator
from rembg import remove, new_session
from tqdm import tqdm

from unidecode import unidecode

import sys, faulthandler, traceback
faulthandler.enable()
sys.setrecursionlimit(3000) 


DATAFIELDS = {
    "dni3": {
        "apellido1": (0.40, 0.30),
        "apellido2": (0.40, 0.35),
        "nombre": (0.40, 0.435),
        "sexo": (0.40, 0.5275),
        "nacionalidad": (0.5375, 0.5275),
        "fecha de nacimiento": (0.4, 0.62),
        "num soport": (0.4, 0.71),
        "validez": (0.58, 0.71),
        "dni": (0.0937, 0.89),
        "can": (0.7859, 0.8361)
    },
    "dni4": {
        "apellido1": (0.395, 0.30),
        "apellido2": (0.395, 0.35),
        "nombre": (0.395, 0.435),
        "sexo": (0.395, 0.5275),
        "nacionalidad": (0.52, 0.5275),
        "fecha de nacimiento": (0.79, 0.5275),
        "num soport": (0.395, 0.71),
        "emision": (0.395, 0.62),
        "validez": (0.59, 0.62),
        "dni": (0.445, 0.185),
        "can": (0.7859, 0.8361)
    },
    "dni3_reverso": {
        "direccion": (0.032, 0.07),
        "nacimiento_poblacion": (0.285, 0.13),
        "nacimiento_provincia": (0.285, 0.18),
        "domicilio_poblacion": (0.28, 0.35),
        "domicilio_provincia": (0.28, 0.4),
        "hijo de": (0.28, 0.54),
    },
    "dni4_reverso": {
        "direccion": (0.285, 0.08),
        "nacimiento_poblacion": (0.285, 0.13),
        "nacimiento_provincia": (0.285, 0.18),
        "domicilio_poblacion": (0.28, 0.35),
        "domicilio_provincia": (0.28, 0.4),
        "hijo de": (0.28, 0.54),
    }
}

mrz = {
    "dni3_reverso": (0.045, 0.7),
    "dni4_reverso": (0.045, 0.7)
}

equipo = {
    "dni3_reverso": (0.04, 0.3),
    "dni4_reverso": (0.04, 0.22)
}

firma = {
    "dni3": (0.45, 0.8),
    "dni4": (0.45, 0.75)
}

num_soport_2 = {
    "dni3": (0.76, 0.175),
    "dni4": (0.775, 0.165),
    "dni4_reverso": (0.125, 0.165)
}

bajorrlieve = {
    "dni3": (0.792, 0.63),
    "dni4": (0.83, 0.67)
}

foto = {
    "dni3": (0.048, 0.1265),
    "dni4": (0.055, 0.32)  #(0.055, 0.231)# "dni4": (0.048, 0.17)
}

tamano_imagen = {
    "dni3": (340, 420),
    "dni4": (360, 460) #"dni4": (410, 500)
}

dni_2 = (0.03, 0.2)

letra_dni = ["T", "R", "W", "A", "G", "M", "Y", "F", "P", "D", "X", "B", "N", "J", "Z", "S", "Q", "V", "H", "L", "C",
             "K", "E"]

HOLOGRAM_PATHS = {
    "dni3": Path("plantillas/holograma_3.png"),
    "dni4": Path("plantillas/holograma_4_nuevo.png"),
}

_HOLOGRAM_CACHE: Dict[Tuple[str, Tuple[int, int]], Image.Image] = {}

UN_MILLON = 1000000
CIEN_MILLONES = 100000000
FONT_PATH = "OCRB Regular/OCRB Regular.ttf"


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def draw_text_90_into(text: str, into, at, font):
    """
    Dibuja texto girado 90 grados hacia la izquierda
    :param text: Texto a escribir
    :param into: Imagen base
    :param at: Punto donde insertar el texto (esquina superior izquierda)
    :param font: Fuente a utilizar
    :return: Imagen con el texto pegado
    """
    # Measure the text area
    boundingbox = font.getbbox(text)
    wi = boundingbox[2] - boundingbox[0] + 4
    hi = boundingbox[3] - boundingbox[1] + 4

    # Copy the relevant area from the source image
    img = Image.new('RGBA', (hi, wi), (255, 255, 255, 0))
    # img = into.crop((at[0], at[1], at[0] + hi, at[1] + wi))

    # Rotate it backwards
    img = img.rotate(270, expand=True)

    # Print into the rotated area
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, font=font, fill=(0, 0, 0, 191), stroke_fill=(0, 0, 0, 88), stroke_width=1)

    # Rotate it forward again
    img = img.rotate(90, expand=True)
    img = img.filter(ImageFilter.BoxBlur(radius=1))

    txt = Image.new('RGBA', into.size, (255, 255, 255, 0))
    # Insert it back into the source image
    # Note that we don't need a mask
    txt.paste(img, at)
    return Image.alpha_composite(into, txt)


def draw_wave_text(draw: ImageDraw.ImageDraw,
                    origin: Tuple[float, float],
                    text: str,
                    base_font_path: str,
                    base_size: int,
                    amplitude: float = 0.20,
                    mirror: bool = False,
                    canvas: Optional[Image.Image] = None,
                    fill=(0, 0, 0, 255),
                    stroke_fill=(0, 0, 0, 255),
                    stroke_width: int = 1) -> None:
    """
    Variante con opción de espejo horizontal (mirror=True) para el reverso.
    Dibuja en un overlay y lo compone en el canvas para poder espejar sin mover coordenadas.
    """
    target = canvas or getattr(draw, "_image", None)
    if target is None:
        return

    n = max(1, len(text))
    mid = (n - 1) / 2.0 if n > 1 else 0.0
    glyphs = []
    total_w = 0.0
    max_h = 0
    for idx, ch in enumerate(text):
        phase = (idx - mid) / max(mid, 1.0)
        scale = 1.0 + amplitude * math.cos(math.pi * phase)
        size = max(1, int(round(base_size * scale)))
        font_var = ImageFont.truetype(base_font_path, size)
        bbox = font_var.getbbox(ch)
        ch_h = bbox[3] - bbox[1]
        total_w += draw.textlength(ch, font=font_var)
        max_h = max(max_h, ch_h)
        glyphs.append((ch, font_var, ch_h))

    overlay = Image.new("RGBA", (int(math.ceil(total_w)) + 2, int(max_h) + 2), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    x = 0.0
    for ch, font_var, ch_h in glyphs:
        y_adj = (max_h - ch_h) * 0.5
        overlay_draw.text(
            (x, y_adj),
            ch,
            font=font_var,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        x += overlay_draw.textlength(ch, font=font_var)

    if mirror:
        overlay = ImageOps.mirror(overlay)

    target.alpha_composite(overlay, dest=(int(origin[0]), int(origin[1])))


def ensure_grayscale_rgba(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    gray = ImageOps.grayscale(image)
    return Image.merge("RGBA", (gray, gray, gray, alpha))


def get_random_fields(apellidos, mujeres, hombres, municipios, data):
    """
    Genera una serie de datos aleatorios de DNI a partir de un conjunto de datos

    :param apellidos: DataFrame con los posibles apellidos
    :param mujeres: DataFrame con los posibles nombres de mujer
    :param hombres: DataFrame con los posibles nombres de hombre
    :param municipios: DataFrame con los posibles municipios
    :param data: Datos de edad y género
    :return: Diccionario con los datos generados
    """
    dni = randrange(CIEN_MILLONES)

    nacimiento = municipios.sample(1).iloc[0]
    domicilio = municipios.sample(1).iloc[0]
    calle = calles.sample(1).iloc[0]

    nombres = mujeres if data["gender"].upper()[0] == 'F' else hombres
    campos = {
        "apellido1": choices(apellidos["apellido"], weights=apellidos["frec_pri"])[0],
        "apellido2": choices(apellidos["apellido"], weights=apellidos["frec_seg"])[0],
        "nombre": choices(nombres["nombre"], weights=nombres["frec"])[0],
        "sexo": data["gender"].upper()[0],
        "nacionalidad": "ESP",
        "fecha de nacimiento": (datetime.datetime.now() - relativedelta(years=data["age"], months=randrange(12),
                                                                        days=randrange(30))).strftime("%d %m %Y"),
        "num soport": "{:c}{:c}{:c}{:06n}".format(randrange(26) + 65, randrange(26) + 65, randrange(26) + 65,
                                                  randrange(UN_MILLON)),
        "emision": datetime.datetime.now().strftime("%d %m %Y"),
        "validez": "PERMANENTE" if data["age"] >= 70 else (
                datetime.datetime.now() + relativedelta(years=5 if data["age"] < 30 else 10)).strftime("%d %m %Y"),
        "dni": "{:08n}{}".format(dni, letra_dni[dni % len(letra_dni)]),
        "can": "{:06n}".format(randrange(UN_MILLON)),
        "equipo": ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(9)),

        "nacimiento_poblacion": unidecode(nacimiento["NOMBRE"]).upper(),
        "nacimiento_provincia": unidecode(nacimiento["provincia"]).upper(),
        "domicilio_poblacion": unidecode(domicilio["NOMBRE"]).upper(),
        "domicilio_provincia": unidecode(domicilio["provincia"]).upper(),
        "direccion": unidecode(" ".join(
            [calle["Clase de la via"].strip(), calle["Particula de la via"].strip(), calle["Nombre de la via"].strip(),
             str(randrange(calle["Numero final del tramo"]) if calle["Numero final del tramo"] > 0 else "S-N")])).upper(),

        "hijo de": "{} / {}".format(choices(hombres["nombre"], weights=hombres["frec"])[0],
                                    choices(mujeres["nombre"], weights=mujeres["frec"])[0])
    }

    return campos


def get_hologram_image(tipo: str, size: Tuple[int, int]) -> Optional[Image.Image]:
    path = HOLOGRAM_PATHS.get(tipo)
    if path is None:
        return None

    path = path if isinstance(path, Path) else Path(path)
    if not path.exists():
        return None

    key = (str(path.resolve()), size)
    hologram = _HOLOGRAM_CACHE.get(key)
    if hologram is None:
        with Image.open(path) as img:
            loaded = img.convert("RGBA")
        if loaded.size != size:
            loaded = loaded.resize(size, Image.LANCZOS)
        _HOLOGRAM_CACHE[key] = loaded
        hologram = loaded
    return hologram.copy()


def make_fake_dni(
    plantilla,
    tipo,
    rembg_session=None,
    visualizer=None,
    visualizer_title: Optional[str] = None,
):
    # --- helpers de visualización ---
    plantilla_pil = Image.fromarray(plantilla)
    visualizer_available = bool(visualizer) and getattr(visualizer, "available", lambda: False)()

    def _label(step: str) -> str:
        return f"{visualizer_title} · {step}" if visualizer_title else step

    def _vis(step: str, image, note: Optional[str] = None) -> None:
        if visualizer_available:
            visualizer.show_step(_label(step), image, note=note)

    def _log(message: str) -> None:
        if visualizer_available:
            visualizer.log(message)

    if visualizer_available and visualizer_title:
        visualizer.section(visualizer_title)

    _vis("Plantilla base", plantilla_pil, "Plantilla cargada desde disco")

    # --- tipografías ---
    font      = ImageFont.truetype(FONT_PATH, plantilla.shape[0] // 20)
    font_big  = ImageFont.truetype(FONT_PATH, plantilla.shape[0] // 13)
    font_small= ImageFont.truetype(FONT_PATH, plantilla.shape[0] // 32)
    font_mrz  = ImageFont.truetype(FONT_PATH, math.floor(plantilla.shape[0] / 12.5))
    font_emision = ImageFont.truetype(FONT_PATH, max(1, plantilla.shape[0] // 20))
    font_sign = ImageFont.truetype("OCRB Regular/PWSignaturetwo.ttf", plantilla.shape[0] // 13)

    # --- datos y foto ---
    face_original = None
    if foto.get(tipo) is not None:
        face_original, face_data = get_random_face()
        if face_data.get("photo_filename"):
            _log(f"Foto seleccionada: {face_data['photo_filename']}")
    else:
        face_data = {"age": randrange(5, 99), "gender": random.choice(["male", "female"]), "photo_filename": ""}

    campos = get_random_fields(apellidos, mujeres, hombres, municipios, face_data)
    _log(", ".join(f"{k}={campos[k]}" for k in ("nombre","apellido1","apellido2","dni","num soport") if k in campos))

    # --- textos sobre lienzo ---
    txt = Image.new("RGBA", plantilla_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)
    for idx, field in DATAFIELDS[tipo].items():
        use_font = font_big if idx in ["dni", "can"] else font
        use_fill = (0, 0, 0, 207)
        use_stroke = (0, 0, 0, 88)
       
        draw.text(
            (field[0] * plantilla.shape[1], field[1] * plantilla.shape[0]),
            campos[idx],
            font=use_font,
            fill=use_fill,
            stroke_width=1,
            stroke_fill=use_stroke,
        )

    if firma.get(tipo) is not None:
        draw.text(
            (firma[tipo][0] * plantilla.shape[1], firma[tipo][1] * plantilla.shape[0]),
            "{} {}{:o}".format(
                campos["nombre"].lower().split(" ")[0], campos["apellido1"].lower(), randrange(0o100)
            ).replace("ñ", "n"),
            font=font_sign, fill=(0, 0, 0, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255),
        )

    if num_soport_2.get(tipo) is not None:
        origin = (
            num_soport_2[tipo][0] * plantilla.shape[1],
            num_soport_2[tipo][1] * plantilla.shape[0],
        )
        draw_wave_text(
            draw,
            origin=origin,
            text=campos["num soport"],
            base_font_path=FONT_PATH,
            base_size=plantilla.shape[0] // 32,
            amplitude=0.25,
            mirror=(tipo == "dni4_reverso"),
            canvas=txt,
            fill=(75, 75, 75, 255),        
            stroke_fill=(90, 90, 90, 255), 
            stroke_width=1,
        )

    if bajorrlieve.get(tipo) is not None:
        draw.text(
            (bajorrlieve[tipo][0] * plantilla.shape[1], bajorrlieve[tipo][1] * plantilla.shape[0]),
            datetime.datetime.now().strftime("%d%m%y"), font=font, fill=(0, 0, 0, 255),
        )

    if equipo.get(tipo) is not None:
        plantilla_pil = draw_text_90_into(
            campos["equipo"],
            plantilla_pil,
            (int(equipo[tipo][0] * plantilla.shape[1]), int(equipo[tipo][1] * plantilla.shape[0])),
            font,
        )

    if mrz.get(tipo) is not None:
        mrz_string = TD1CodeGenerator(
            "ID",
            "ESP",
            campos["num soport"],
            datetime.datetime.strptime(campos["fecha de nacimiento"], "%d %m %Y").strftime("%y%m%d"),
            campos["sexo"],
            ("991231" if campos["validez"] == "PERMANENTE"
             else datetime.datetime.strptime(campos["validez"], "%d %m %Y").strftime("%y%m%d")),
            campos["nacionalidad"], "", "", optional_data1=campos["dni"],
        )
        lines = str(mrz_string).split("\n")
        lines[2] = "{}<{}<<{}{}".format(campos["apellido1"], campos["apellido2"], campos["nombre"], "<"*30)\
                     .replace(" ", "<")[:30]
        draw.text(
            (mrz[tipo][0] * plantilla.shape[1], mrz[tipo][1] * plantilla.shape[0]),
            "\n".join(lines), font=font_mrz, fill=(0, 0, 0, 255), spacing=plantilla.shape[1] // 60,
        )

    if tipo == "dni4":
        draw.text(
            (dni_2[0] * plantilla.shape[1], dni_2[1] * plantilla.shape[0]),
            campos["dni"], font=font_small, fill=(0, 0, 0, 255),
        )

    # emborrón leve y composición de textos
    txt = txt.filter(ImageFilter.BoxBlur(radius=1))
    image = img_to_array(plantilla_pil)
    plantilla = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    plantilla_pil = Image.alpha_composite(plantilla_pil, txt)
    _vis("Campos impresos", plantilla_pil)

    # --- Foto principal con encaje anti-corte de cabeza ---
    if foto.get(tipo) is not None:
        x_offset = int(foto[tipo][0] * plantilla.shape[1])
        y_offset = int((foto[tipo][1] + 0.015) * plantilla.shape[0])  # baja ~1.5% de la altura total


        if face_original is not None:
            _vis("Foto original", Image.fromarray(cv2.cvtColor(face_original, cv2.COLOR_BGR2RGB)),
                 face_data.get("photo_filename"))

        # Segmentación fondo
        face_rgba = remove(face_original, session=rembg_session)

        # recorte lateral según QA
        cl, cr = int(face_data.get("cut_left", 0)), int(face_data.get("cut_right", 0))
        face_rgba = face_rgba[:, cl: (None if cr <= 0 else -cr)]

        # preprocesado (gris, suavizado, alfa refinada)
        face_pil, face_small_pil = process_face(face_rgba)

        w, h = tamano_imagen[tipo]
        fitted_pil = fit_face_with_headroom(
            np.array(face_pil), (w, h),
            headroom_ratio=0.21,     
            side_margin_ratio=0.04,  
            anchor_y=0.58,
            zoom=1.28               
        )

        print("DEBUG fit:", fitted_pil.size, "window:", (w,h))



        # máscara redondeada + borde limpio
        # máscara redondeada: radio moderado para evitar picos en las esquinas
        radius_px = int(min(w, h) * 0.08)
        try:
            mask_img = rounded_rect_mask_aa(w, h, r=radius_px)
        except NameError:    
            mask_img = rounded_rect_mask(w, h, r=radius_px)

        fitted_np = np.array(fitted_pil)
        mask_np   = np.array(mask_img, dtype=np.uint8)
        alpha_new = (fitted_np[:, :, 3].astype(np.uint16) * mask_np.astype(np.uint16) // 255).astype(np.uint8)
        alpha_new[alpha_new < 14] = 0
        fitted_np[:, :, 3] = alpha_new
        face_window = Image.fromarray(fitted_np, "RGBA")

        # sombra + pegado
        sombra = soft_drop_shadow((w, h), radius=14, opacity=90, offset=(5, 7))
        plantilla_pil.alpha_composite(sombra, (x_offset - 2, y_offset - 2 + 6))
        plantilla_pil.alpha_composite(face_window, (x_offset, y_offset + 6))


        # retrato “fantasma” inferior-derecha
        plantilla_pil.alpha_composite(
            face_small_pil,
            (int((bajorrlieve[tipo][0] + 0.025) * plantilla.shape[1]),
             int((bajorrlieve[tipo][1] - 0.05)  * plantilla.shape[0]))
        )

        _vis("Foto integrada", plantilla_pil)

    # --- holograma (si existe) ---
    hologram_image = get_hologram_image(tipo, plantilla_pil.size)
    if hologram_image is not None:
        plantilla_pil = Image.alpha_composite(plantilla_pil, hologram_image)
        _vis("Holograma aplicado", plantilla_pil)

    # --- salida final (BGR para OpenCV) ---
    result = img_to_array(plantilla_pil)
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    _vis("DNI finalizado", result, "Documento sintetizado listo")

    return result, campos, face_data["photo_filename"]



def process_face(face):
    """
    Cara en estilo DNI:
    - Desatura / corrige luminancia suave
    - Gris + contraste con CLAHE (más natural que equalizeHist)
    - Borde con feather corto (sin comer pelo)
    - Genera retrato pequeño
    """
    alpha_mask = face[:, :, 3].astype(np.uint8)
    bgr = face[:, :, :3].copy()

    # Color: desaturación y luminancia suave
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.70   
    hsv[:, :, 2] *= 0.98
    bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Suavizado piel muy leve
    bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=30, sigmaSpace=30)

    # Gris + contraste suave con CLAHE (evita “bordes duros”)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)

    # Construir RGBA
    rgba = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGBA)

    # filtro para mejorar transparencia de fondo
    a = refine_alpha(alpha_mask, shrink=3, blur=5, open_iter=1)
    rgba[:, :, 3] = np.clip(a.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
    alpha_norm = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
    rgba[:, :, :3] = (rgba[:, :, :3].astype(np.float32) * alpha_norm).astype(np.uint8)

    face_pil = Image.fromarray(rgba, mode="RGBA")

    # Retrato pequeño (bajorrelieve)
    small = cv2.resize(rgba, (68, 86))
    small[:, :, 3] = (small[:, :, 3].astype(np.float32) * 0.45).astype(np.uint8)
    face_small_pil = Image.fromarray(small, mode="RGBA")

    return face_pil, face_small_pil



def adjust_color_levels(face, blacks, gamma, whites, blacks_out=0, whites_out=0):
    """
    Ajusta los niveles de brillo de una imagen similarmente a herramientas como Photoshop o GIMP
    :param face:
    :param blacks:
    :param gamma:
    :param whites:
    :param blacks_out:
    :param whites_out:
    :return:
    """
    in_black = np.array([blacks, blacks, blacks], dtype=np.float32)
    in_white = np.array([whites, whites, whites], dtype=np.float32)
    in_gamma = np.array([gamma, gamma, gamma], dtype=np.float32)
    out_black = np.array([blacks_out, blacks_out, blacks_out], dtype=np.float32)
    out_white = np.array([whites_out, whites_out, whites_out], dtype=np.float32)
    merged_img = np.clip((face - in_black) / (in_white - in_black), 0, 255)
    merged_img = (merged_img ** (1 / in_gamma)) * (out_white - out_black) + out_black
    merged_img = np.clip(merged_img, 0, 255).astype(np.uint8)
    return merged_img


if __name__ == "__main__":
    model_name = "u2net_human_seg"
    session = new_session(model_name)

    # Abrimos los conjuntos de datos
    apellidos = pd.read_csv("data/apellidos.csv")
    mujeres = pd.read_csv("data/mujeres.csv")
    hombres = pd.read_csv("data/hombres.csv")
    provincias = pd.read_csv("data/provinces_es.csv")
    municipios = pd.read_csv("data/11codmun.csv", sep=";", skiprows=1)
    calles = pd.read_csv("data/CALLEJERO_VIGENTE_TRAMERO_202304.csv", sep=";", encoding="latin-1")

    # Datos del repositorio de fotos preanalizadas
    fotos = pd.read_csv("photo_qa_synthesis.csv", sep=";")
    fotos = fotos.sort_values(["score"], ascending=False)

    # apellidos = apellidos[:1000]
    # mujeres = mujeres[:1000]
    # hombres = hombres[:1000]
    fotos = fotos[:2000].sample(frac=1, random_state=42).reset_index(drop=True)

    # Añade el nombre de provincia a la tabla de municipios
    municipios["provincia"] = municipios.apply(
        lambda row: provincias.loc[provincias["postal_code"] == row["CPRO"]]["name"].iloc[0], axis=1)
    tipo = "dni3"

    # Lee las plantillas de los diferentes tipos de DNI
    plantilla = {}
    # plantilla["dni3"] = cv2.imread("plantillas/anverso_3_planZ
    plantilla["dni4"] = cv2.imread("plantillas/anverso_4_nuevo.png")
    plantilla["dni4"] = cv2.cvtColor(plantilla["dni4"], cv2.COLOR_BGR2RGBA)
    # plantilla["dni3_reverso"] = cv2.imread("plantillas/reverso_3_plantilla.png")
    # plantilla["dni3_reverso"] = cv2.cvtColor(plantilla["dni3_reverso"], cv2.COLOR_BGR2RGBA)
    plantilla["dni4_reverso"] = cv2.imread("plantillas/reverso_4_nuevo.png")
    plantilla["dni4_reverso"] = cv2.cvtColor(plantilla["dni4_reverso"], cv2.COLOR_BGR2RGBA)

    debug_mode = True
    if debug_mode:
        result, data, photo = make_fake_dni(plantilla[tipo], tipo, session)

        cv2.imshow("result", result)
        print(data)
        while cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) > 0:
            if cv2.waitKey(50) >= 0:
                break

    else:
        dir = "C:\\Users\\kamoresc\\Documents\\datasets\\dnis_falsos"

        outfile = open("generated_dnis.csv", "w+")
        outfile.close()
        columns = None
        outfile = open("generated_dnis.csv", "a+")

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        for tipo in plantilla.keys():
            if not os.path.exists(os.path.join(dir, tipo)):
                os.makedirs(os.path.join(dir, tipo), exist_ok=True)
            for i in tqdm(range(1000), desc="Generating {} ({}/{})".format(tipo, list(plantilla.keys()).index(tipo) + 1,
                                                                           len(plantilla))):
                #result, data, photo = make_fake_dni(plantilla[tipo], tipo, session)

                try:
                  result, data, photo = make_fake_dni(plantilla[tipo], tipo, session)
                except Exception as e:
                 print("ERROR:", repr(e))
                 traceback.print_exc()
                 raise

                cv2.imwrite(os.path.join(dir, tipo, "{}.png".format(i)), result)

                data = {
                    "filename": os.path.join(dir, tipo, "{}.png".format(i)),
                    "photo_filename": photo,
                    "type": tipo,
                    **data
                }

                if columns is None:
                    columns = data.keys()
                    outfile.write(";".join(columns))
                    outfile.write("\n")

                outfile.write(";".join(data.values()))
                outfile.write("\n")


# New utils

def rounded_rect_mask(w, h, r):
    """Máscara [0..255] con esquinas redondeadas (radio r)."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    # Suaviza un poco el borde
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, r // 8)))

def soft_drop_shadow(size, radius=12, opacity=95, offset=(5, 7)):
    """Sombra suave SOLO alrededor (interior totalmente transparente)."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = size
    # alpha local del tamaño de la foto
    alpha = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(alpha)
    d.rounded_rectangle([0, 0, w, h], radius=radius, fill=opacity)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))

    # vaciamos el interior: no “lava” el fondo nunca
    d = ImageDraw.Draw(alpha)
    d.rounded_rectangle([0, 0, w, h], radius=radius, fill=0)

    W, H = w + offset[0], h + offset[1]
    a_canvas = Image.new("L", (W, H), 0)
    a_canvas.paste(alpha, offset)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(a_canvas)
    return shadow

def _mask_bbox(alpha: np.ndarray, thr: int = 10):
    ys, xs = np.where(alpha > thr)
    if len(xs) == 0 or len(ys) == 0:
        h, w = alpha.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1

def fit_face_with_headroom(
    face_rgba: np.ndarray,
    win_size: tuple[int, int],
    headroom_ratio: float = 0.12,
    side_margin_ratio: float = 0.06,
    anchor_y: float = 0.60,
    zoom: float = 1.0,
    allow_crop: bool = True,
) -> Image.Image:
    """
    Encaja la foto en una ventana (w,h) sin cortar la cabeza:
      - Recorta primero el contorno útil para evitar que el rostro quede en una esquina.
      - Respeta holguras configurables antes de escalar.
      - Centra el bbox resultante en anchor_y y pega usando intersecciones seguras.
    Devuelve un PIL RGBA de tamaño exacto (w,h).
    """
    w, h = win_size
    alpha = face_rgba[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if ys.size == 0:
        return Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), "RGBA")

    # 1) BBOX inicial + margen antes de escalar (evita desplazamientos extremos)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    pad_x = int(round(bw * 0.08))
    pad_y = int(round(bh * 0.12))

    H, W = face_rgba.shape[:2]
    x0 = max(0, x0 - pad_x)
    x1 = min(W - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(H - 1, y1 + pad_y)

    face_rgba = face_rgba[y0: y1 + 1, x0: x1 + 1]
    crop_alpha = face_rgba[:, :, 3]
    ys, xs = np.where(crop_alpha > 16)
    if ys.size == 0:
        return Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), "RGBA")

    cbw = int(xs.max() - xs.min() + 1)
    cbh = int(ys.max() - ys.min() + 1)

    Hc, Wc = face_rgba.shape[:2]
    if cbw < 0.25 * Wc or cbh < 0.25 * Hc:
        cbw, cbh = Wc, Hc

    # 2) Holguras y escala
    headroom = int(headroom_ratio * h)
    side_m   = int(side_margin_ratio * w)
    bottom_m = int(0.05 * h)

    fit_w = max(1, w - 2 * side_m)
    fit_h = max(1, h - headroom - bottom_m)

    base_scale = min(fit_w / cbw, fit_h / cbh)   # escala que cabe exacta
    scale = base_scale * max(1.0, float(zoom))   # “zoom” por encima de la que cabe

    if not allow_crop:
        # si no quieres recorte, no permitas exceder base_scale
        scale = min(scale, base_scale)

    scale = float(np.clip(scale, 1e-3, 6.0))     # límite superior prudente

    new_W = max(1, int(round(Wc * scale)))
    new_H = max(1, int(round(Hc * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    scaled = cv2.resize(face_rgba, (new_W, new_H), interpolation=interp)

    a_scaled = scaled[:, :, 3]
    ys2, xs2 = np.where(a_scaled > 16)
    sx0, sx1 = int(xs2.min()), int(xs2.max())
    sy0, sy1 = int(ys2.min()), int(ys2.max())
    sbw, sbh = (sx1 - sx0 + 1), (sy1 - sy0 + 1)

    # 3) Anclaje del bbox en la ventana final
    cx = w // 2
    cy = int(anchor_y * h)
    desired_left_bbox = cx - sbw // 2
    desired_top_bbox  = cy - sbh // 2

    left_bbox = int(np.clip(desired_left_bbox, side_m, w - side_m - sbw))
    top_bbox  = int(np.clip(desired_top_bbox,  headroom,  h - bottom_m - sbh))

    left_img = left_bbox - sx0
    top_img  = top_bbox  - sy0

    # 4) Pegado robusto (intersección)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    dx0 = max(0, left_img)
    dy0 = max(0, top_img)
    dx1 = min(w, left_img + new_W)
    dy1 = min(h, top_img  + new_H)

    if dx1 > dx0 and dy1 > dy0:
        sx0_i = dx0 - left_img
        sy0_i = dy0 - top_img
        sx1_i = sx0_i + (dx1 - dx0)
        sy1_i = sy0_i + (dy1 - dy0)

        src = scaled[sy0_i:sy1_i, sx0_i:sx1_i].astype(np.float32)
        dst = canvas[dy0:dy1, dx0:dx1].astype(np.float32)

        alpha = (src[:, :, 3:4] / 255.0)
        inv_a = 1.0 - alpha
        canvas[dy0:dy1, dx0:dx1, :3] = alpha * src[:, :, :3] + inv_a * dst[:, :, :3]
        canvas[dy0:dy1, dx0:dx1,  3] = np.clip(src[:, :, 3] + dst[:, :, 3] * inv_a[:, :, 0], 0, 255)

    return Image.fromarray(canvas, "RGBA")


def recenter_vert(pil_rgba: Image.Image, target_y=0.56, max_shift_px=10) -> Image.Image:
    """
    Recentra verticalmente la imagen (por la máscara alfa) para que el rostro quede
    alineado según el centro de masa de los píxeles visibles.
    target_y = posición vertical deseada (0 = arriba, 1 = abajo)
    max_shift_px = desplazamiento máximo permitido.
    """
    import numpy as np
    arr = np.array(pil_rgba)
    if arr.shape[2] < 4:
        return pil_rgba  # sin canal alfa, no hay nada que centrar

    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if ys.size == 0:
        return pil_rgba

    # Centro vertical de la máscara
    cy = ys.mean() / alpha.shape[0]
    # Diferencia respecto al objetivo
    dy = int(np.clip((target_y - cy) * alpha.shape[0], -max_shift_px, max_shift_px))

    if dy != 0:
        arr = np.roll(arr, dy, axis=0)
        # Limpia los bordes desplazados
        if dy > 0:
            arr[:dy, :, :] = 0
        else:
            arr[dy:, :, :] = 0

    return Image.fromarray(arr, "RGBA")

def refine_alpha(alpha: np.ndarray,
                 shrink: int = 1,   # 0 o 1 px de erosión
                 blur: int   = 3,   # 3 = feather suave
                 open_iter: int = 0 # 0–1 para limpiar puntitos
                 ) -> np.ndarray:
    """
    Limpia y suaviza la máscara sin 'comerse' pelo/orejas.
    - shrink: erosión en px (0 o 1 recomendable)
    - blur:   radio gaussiano impar (1,3,5...). 3 es buen punto.
    - open_iter: apertura morfológica ligera para artefactos sueltos
    """
    a = alpha.astype(np.uint8)
    # binaria estable
    _, a = cv2.threshold(a, 8, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if open_iter > 0:
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, k, iterations=open_iter)

    if shrink > 0:
        a = cv2.erode(a, k, iterations=shrink)

    # feather corto en el borde (no lo hagas muy grande)
    if blur and blur > 0:
        if blur % 2 == 0:
            blur += 1
        a = cv2.GaussianBlur(a, (blur, blur), 0)

    return a
