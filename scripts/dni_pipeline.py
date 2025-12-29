"""
Dni Pipeline
=================
Este script implementa el flujo de trabajo (pipeline) completo 
para generar un DNI desde cero. Su función principal es coordinar 
la creación de la foto por IA, inyectar esa foto y los datos de la persona 
en el motor de generación de documentos (Synthetic Data), y guardar los resultados 
finales (anverso, reverso y metadatos) de forma estructurada.
"""

__author__ = "Antonio Calvo"
__copyright__ = "Copyright 2025, Universidad de Extremadura"
__credits__ = ["Antonio Calvo", "Fernando Broncano", "Sergio Guijarro"]
__version__ = "1.0.1"
__maintainer__ = "Antonio Calvo"
__email__ = "acalvopic@unex.es"
__status__ = "Development"

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Callable, Iterable, Optional, TYPE_CHECKING
from random import randrange, random

import cv2
import pandas as pd
from rembg import new_session

import scripts.synthetic_data as generator
from scripts.ImageGeneration import generate as generate_ai_image

if TYPE_CHECKING:
    from scripts.visualization import DniVisualizer

DEFAULT_OUTPUT_DIR = Path("salidas")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_OUTPUT_DIR / "numba_cache"))
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
(DEFAULT_OUTPUT_DIR / "numba_cache").mkdir(parents=True, exist_ok=True)

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _noop(*_args, **_kwargs):
        return None

    pyplot_stub.plot = _noop
    pyplot_stub.xlabel = _noop
    pyplot_stub.ylabel = _noop
    pyplot_stub.show = _noop

    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub


class FaceDatasetError(RuntimeError):
    """Raised when there is a problem with the faces dataset."""


def ensure_rgba(image_path: Path) -> "cv2.typing.MatLike":
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FaceDatasetError(f"No se pudo leer la plantilla: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    raise FaceDatasetError(f"Formato de imagen no soportado para {image_path}")


def generate_random_person_data() -> dict:
    """
    Genera datos aleatorios de una persona (edad, género).
    
    Returns:
        dict: {"age": int, "gender": str}
    """
    age = randrange(18, 80)  # Edad entre 18 y 80 años
    gender = "male" if random() < 0.5 else "female"
    
    return {
        "age": age,
        "gender": gender
    }


def generate_ai_face_image(
    person_data: dict,
    output_dir: Path,
    label: str = "dni",
    visualizer: Optional["DniVisualizer"] = None,
) -> Optional[Path]:
    """
    Genera una imagen de rostro usando IA basándose en los datos de la persona.
    
    Args:
        person_data: Diccionario con 'age' y 'gender'
        output_dir: Directorio donde guardar la imagen
        label: Prefijo para el nombre del archivo
        visualizer: Visualizador opcional
        
    Returns:
        Path a la imagen generada o None si falla
    """
    def _log(message: str):
        if visualizer and visualizer.available():
            visualizer.log(message)
        else:
            print(message)
    
    _log(f"Generando imagen de {person_data['gender']}, {person_data['age']} años con IA...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{label}_ai_face.png"
    
    try:
        image_path = generate_ai_image(
            gender=person_data['gender'],
            age=person_data['age'],
            output_dir=str(output_dir),
            output_name=output_name
        )
        
        if image_path and Path(image_path).exists():
            _log(f"✓ Imagen generada correctamente: {image_path}")
            return Path(image_path)
        else:
            _log("✗ Error: La generación de imagen no devolvió una ruta válida")
            return None
            
    except Exception as e:
        _log(f"✗ Error al generar imagen con IA: {e}")
        return None


def prepare_generator_state_minimal() -> None:
    """Prepara el estado del generador sin necesidad de dataset de caras."""
    generator.apellidos = pd.read_csv("data/apellidos.csv")
    generator.mujeres = pd.read_csv("data/mujeres.csv")
    generator.hombres = pd.read_csv("data/hombres.csv")
    generator.provincias = pd.read_csv("data/provinces_es.csv")
    generator.municipios = pd.read_csv("data/11codmun.csv", sep=";", skiprows=1)
    generator.calles = pd.read_csv(
        "data/CALLEJERO_VIGENTE_TRAMERO_202304.csv", sep=";", encoding="latin-1"
    )

    generator.municipios["provincia"] = generator.municipios.apply(
        lambda row: generator.provincias.loc[
            generator.provincias["postal_code"] == row["CPRO"], "name"
        ].iloc[0],
        axis=1,
    )
    
    # No necesitamos fotos porque las generaremos con IA
    generator.fotos = None
    generator.debug_mode = False


def create_face_dataframe_from_ai_image(image_path: Path) -> pd.DataFrame:
    """
    Crea un DataFrame temporal con la información de la imagen generada por IA.
    
    Args:
        image_path: Ruta a la imagen generada
        
    Returns:
        DataFrame con formato compatible con el pipeline original
    """
    # Las imágenes de IA vienen con fondo blanco, así que no necesitamos recortes
    return pd.DataFrame([{
        "filename": str(image_path),
        "score": 100.0,
        "cut_left": 0,
        "cut_right": 0
    }])


def render_with_campos_and_image(
    plantilla: "cv2.typing.MatLike",
    tipo: str,
    session,
    face_image_path: Path,
    person_data: dict,
    campos_override: Optional[dict] = None,
    visualizer: Optional["DniVisualizer"] = None,
    visualizer_title: Optional[str] = None,
) -> tuple["cv2.typing.MatLike", dict, str]:
    """
    Renderiza el DNI usando la imagen generada y datos proporcionados.
    """
    # Configurar el generador para usar nuestra imagen específica
    generator.fotos = create_face_dataframe_from_ai_image(face_image_path)
    
    # Si no hay campos override, los generamos basándonos en person_data
    if campos_override is None:
        campos_override = generator.get_random_fields(
            generator.apellidos,
            generator.mujeres,
            generator.hombres,
            generator.municipios,
            person_data
        )
    
    # Parchear temporalmente get_random_fields y get_random_face
    original_get_random_fields = generator.get_random_fields
    
    def _patched_get_random_fields(*_args, **_kwargs):
        return campos_override.copy()
    
    def _patched_get_random_face():
        """Devuelve la imagen generada por IA con los datos correctos."""
        img = cv2.imread(str(face_image_path))
        if img is None:
            raise FaceDatasetError(f"No se pudo leer la imagen generada: {face_image_path}")
        
        return img, {
            "age": person_data["age"],
            "gender": person_data["gender"],
            "cut_left": 0,
            "cut_right": 0,
            "photo_filename": str(face_image_path)
        }
    
    generator.get_random_fields = _patched_get_random_fields
    generator.get_random_face = _patched_get_random_face
    
    try:
        result, campos, photo_filename = generator.make_fake_dni(
            plantilla,
            tipo,
            session,
            visualizer=visualizer,
            visualizer_title=visualizer_title,
        )
        return result, campos, photo_filename
    finally:
        generator.get_random_fields = original_get_random_fields


def generate_dni_pair_with_ai(
    session,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    person_data: Optional[dict] = None,
    campos_override: Optional[dict] = None,
    front_template: Path = Path("plantillas/anverso_4_plantilla.png"),
    back_template: Path = Path("plantillas/reverso_4_plantilla.png"),
    label: str = "dni",
    visualizer: Optional["DniVisualizer"] = None,
    keep_ai_image: bool = False,
) -> dict:
    """
    Genera un DNI completo usando IA para crear la foto.
    
    Args:
        session: Sesión de rembg
        output_dir: Directorio de salida
        person_data: Datos de la persona (age, gender). Si es None, se generan aleatorios
        campos_override: Campos del DNI a usar en vez de generarlos
        front_template: Plantilla del anverso
        back_template: Plantilla del reverso
        label: Prefijo para los archivos
        visualizer: Visualizador opcional
        keep_ai_image: Si True, mantiene la imagen generada por IA
        
    Returns:
        dict con rutas a los archivos generados y metadatos
    """
    # 1. Generar datos de persona si no se proporcionan
    if person_data is None:
        person_data = generate_random_person_data()
        if visualizer and visualizer.available():
            visualizer.log(f"Datos generados: {person_data['gender']}, {person_data['age']} años")
    
    # 2. Generar imagen con IA
    ai_image_path = generate_ai_face_image(
        person_data,
        output_dir,
        label,
        visualizer
    )
    
    if ai_image_path is None:
        raise FaceDatasetError("No se pudo generar la imagen con IA")
    
    # 3. Cargar plantillas
    front_template_img = ensure_rgba(front_template)
    back_template_img = ensure_rgba(back_template)
    
    # 4. Renderizar anverso
    front_image, campos, photo_filename = render_with_campos_and_image(
        front_template_img,
        "dni4",
        session,
        ai_image_path,
        person_data,
        campos_override,
        visualizer=visualizer,
        visualizer_title="Anverso",
    )
    
    # 5. Renderizar reverso
    back_image, _, _ = render_with_campos_and_image(
        back_template_img,
        "dni4_reverso",
        session,
        ai_image_path,
        person_data,
        campos_override or campos,
        visualizer=visualizer,
        visualizer_title="Reverso",
    )
    
    # 6. Guardar resultados
    output_dir.mkdir(parents=True, exist_ok=True)
    front_path = output_dir / f"{label}_anverso.png"
    back_path = output_dir / f"{label}_reverso.png"
    
    if not cv2.imwrite(str(front_path), front_image):
        raise FaceDatasetError(f"No se pudo guardar el anverso en {front_path}")
    if not cv2.imwrite(str(back_path), back_image):
        raise FaceDatasetError(f"No se pudo guardar el reverso en {back_path}")
    
    # 7. Guardar metadatos
    metadata_path = output_dir / f"{label}_campos.json"
    metadata = campos.copy()
    metadata["photo_filename"] = str(ai_image_path)
    metadata["ai_generated"] = True
    metadata["person_data"] = person_data
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    # 8. Limpiar imagen temporal si no se quiere mantener
    if not keep_ai_image and ai_image_path.exists():
        ai_image_path.unlink()
    
    return {
        "front": front_path,
        "back": back_path,
        "metadata": metadata_path,
        "campos": campos,
        "person_data": person_data,
        "ai_image_path": ai_image_path if keep_ai_image else None,
    }


def new_rembg_session(model_name: str = "u2net_human_seg"):
    return new_session(model_name)


__all__ = [
    "FaceDatasetError",
    "DEFAULT_OUTPUT_DIR",
    "prepare_generator_state_minimal",
    "generate_dni_pair_with_ai",
    "generate_random_person_data",
    "new_rembg_session",
]