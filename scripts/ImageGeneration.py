"""
Image Generation
=================
Su función es conectar el modelo de generaación de 
imágenes para generar retratos fotorrealistas de estilo biométrico/pasaporte, 
basados en una edad y género específicos.

Se utiliza una API REST para comunicarse con el modelo de generación de imágenes.

En caso de tener desplegado el modelo, ajustar URL.
"""

__author__ = "Antonio Calvo"
__copyright__ = "Copyright 2025, Universidad de Extremadura"
__credits__ = ["Antonio Calvo", "Fernando Broncano", "Sergio Guijarro"]
__version__ = "1.0.1"
__maintainer__ = "Antonio Calvo"
__email__ = "acalvopic@unex.es"
__status__ = "Development"

import requests
import json
import os
import argparse
from typing import Optional

URL = "http://127.0.0.1:8888/v1/generation/text-to-image"


def build_prompt(gender: str, age: int) -> str:
    """Builds the prompt text based on gender and age."""
    subject_desc = f"{age}-year-old {gender}"
    return (
        f"raw photo, biometric passport photo of a {subject_desc}, medium shot framing (chest up), "
        "ensuring entire head and full hair are completely visible with ample empty space above and around the head, "
        "uncropped, centered, front view, looking directly at camera, perfectly neutral expression, mouth closed, "
        "plain solid white background, flat even studio lighting, high detailed natural skin texture, "
        "sharp focus, dslr, 8k uhd"
    )


def generate(
    gender: str = "female",
    age: int = 30,
    output_dir: str = "imagenes_generadas",
    output_name: str = "resultado.png",
) -> Optional[str]:
    """
    Generates a DNI/passport image.

    Returns:
        Path to saved image if successful, otherwise None.
    """

    prompt_text = build_prompt(gender, age)

    payload = {
        "prompt": prompt_text,
        "negative_prompt": (
            "(cropped:1.5), (out of frame:1.5), (cut off head:1.5), (cut off hair:1.5), "
            "forehead cut off, chin cut off, extreme close up, tight framing, macro shot, "
            "smiling, open mouth, teeth showing, looking away, shadows on face, blurry, low quality"
        ),
        "style_selections": [
            "Fooocus V2",
            "Fooocus Enhance",
            "Fooocus Sharp"
        ],
        "performance_selection": "Speed",
        "aspect_ratios_selection": "896*1152",
        "image_number": 1,
        "image_seed": -1,
        "sharpness": 2,
        "guidance_scale": 4,
        "base_model_name": "juggernautXL_v8Rundiffusion.safetensors",
        "refiner_model_name": "None",
        "refiner_switch": 0.5,
        "loras": [
            {"enabled": True, "model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.1},
        ],
        "advanced_params": {
            "sampler_name": "dpmpp_2m_sde_gpu",
            "scheduler_name": "karras",
            "clip_skip": 2,
            "vae_name": "Default (model)"
        },
        "save_meta": True,
        "save_extension": "png",
        "require_base64": False,
        "async_process": False
    }

    headers = {
        "Accept": "image/png",
        "Content-Type": "application/json"
    }

    response = requests.post(URL, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        print("Error:", response.text)
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate DNI/passport photo using gender and age"
    )
    parser.add_argument("--gender", "-g", default="female", help="Gender (male, female)")
    parser.add_argument("--age", "-a", type=int, default=30, help="Age in years")
    parser.add_argument("--out", "-o", default="resultado.png", help="Output image name")

    args = parser.parse_args()

    path = generate(
        gender=args.gender,
        age=args.age,
        output_name=args.out,
    )

    if path:
        print(f"Image saved at: {path}")


if __name__ == "__main__":
    main()