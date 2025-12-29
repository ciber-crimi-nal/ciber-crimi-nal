from __future__ import annotations
"""
Main
=================
Este script actúa como el punto de entrada principal (CLI) para la 
generación de DNIs españoles sintéticos. Su objetivo es crear lotes de imágenes 
de documentos de identidad (anverso y reverso) utilizando Inteligencia Artificial 
para generar rostros realistas que coincidan con los metadatos (edad, género) 
especificados o aleatorios.
"""

__author__ = "Antonio Calvo"
__copyright__ = "Copyright 2025, Universidad de Extremadura"
__credits__ = ["Antonio Calvo", "Fernando Broncano", "Sergio Guijarro"]
__version__ = "1.0.1"
__maintainer__ = "Antonio Calvo"
__email__ = "acalvopic@unex.es"
__status__ = "Development"

import argparse
from pathlib import Path

from scripts.dni_pipeline import (
    DEFAULT_OUTPUT_DIR,
    FaceDatasetError,
    generate_dni_pair_with_ai,
    generate_random_person_data,
    new_rembg_session,
    prepare_generator_state_minimal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI para generar DNIs sintéticos usando IA para las fotos.",
    )
    parser.add_argument(
        "--age",
        type=int,
        default=None,
        help="Edad de la persona (si no se especifica, se genera aleatoriamente).",
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female"],
        default=None,
        help="Género de la persona (male/female). Si no se especifica, se genera aleatoriamente.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio donde se guardarán las imágenes (por defecto {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--front-template",
        type=Path,
        default=Path("plantillas/anverso_4_nuevo.png"),
        help="Plantilla a usar para el anverso.",
    )
    parser.add_argument(
        "--back-template",
        type=Path,
        default=Path("plantillas/reverso_4_plantilla.png"),
        help="Plantilla a usar para el reverso.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Número de DNIs a generar.",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="dni",
        help="Prefijo para nombrar los ficheros resultantes.",
    )
    parser.add_argument(
        "--keep-ai-images",
        action="store_true",
        help="Mantiene las imágenes generadas por IA (por defecto se eliminan).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Muestra una ventana con el proceso de creación paso a paso.",
    )
    parser.add_argument(
        "--visualize-delay",
        type=float,
        default=0.6,
        help="Segundos de pausa entre pasos durante la visualización (por defecto 0.6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Preparar el generador (sin necesidad de dataset de caras)
    print("[INFO] Preparando generador...")
    prepare_generator_state_minimal()
    
    # Inicializar sesión de rembg
    print("[INFO] Inicializando rembg...")
    session = new_rembg_session()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configurar visualizador si se solicita
    visualizer = None
    if args.visualize:
        try:
            from scripts.visualization import DniVisualizer, VisualizerConfig

            config = VisualizerConfig(
                delay_seconds=max(0.0, args.visualize_delay),
            )
            visualizer = DniVisualizer(config=config)
            if not visualizer.available():
                print("[WARN] No se pudo iniciar la visualización (¿entorno sin Tkinter?).")
                visualizer = None
        except Exception as exc:
            print(f"[WARN] No se pudo iniciar la visualización: {exc}")
            visualizer = None

    # Generar DNIs
    for index in range(args.count):
        label = args.label_prefix
        if args.count > 1:
            label = f"{args.label_prefix}_{index:03d}"

        # Preparar datos de persona
        person_data = None
        if args.age is not None or args.gender is not None:
            person_data = generate_random_person_data()
            if args.age is not None:
                person_data["age"] = args.age
            if args.gender is not None:
                person_data["gender"] = args.gender
        
        print(f"\n[INFO] Generando DNI {index + 1}/{args.count}...")
        if person_data:
            print(f"       Persona: {person_data['gender']}, {person_data['age']} años")

        try:
            result = generate_dni_pair_with_ai(
                session,
                output_dir=args.output_dir,
                person_data=person_data,
                label=label,
                front_template=args.front_template,
                back_template=args.back_template,
                visualizer=visualizer,
                keep_ai_image=args.keep_ai_images,
            )
            
            print(f"[OK] Generado {label}:")
            print(f"     Anverso: {result['front']}")
            print(f"     Reverso: {result['back']}")
            print(f"     Metadatos: {result['metadata']}")
            if result.get('ai_image_path'):
                print(f"     Imagen IA: {result['ai_image_path']}")
                
        except Exception as e:
            print(f"[ERROR] No se pudo generar el DNI {label}: {e}")
            if args.count == 1:
                raise
            continue

    if visualizer is not None and visualizer.available():
        visualizer.close()

    print("\n[INFO] Proceso completado.")


if __name__ == "__main__":
    try:
        main()
    except (FaceDatasetError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}")