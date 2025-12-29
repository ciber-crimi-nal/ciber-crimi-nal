"""
Visualization
=================
Este script proporciona una interfaz gráfica de usuario (GUI) 
ligera para observar el proceso de generación del DNI en tiempo real. 
Actúa como una ventana de depuración o demostración, permitiendo al usuario 
ver cómo el script Synthetic Data va componiendo las capas (fondo, textos, foto, firma) 
paso a paso.
"""

__author__ = "Antonio Calvo"
__copyright__ = "Copyright 2025, Universidad de Extremadura"
__credits__ = ["Antonio Calvo", "Fernando Broncano", "Sergio Guijarro"]
__version__ = "1.0.1"
__maintainer__ = "Antonio Calvo"
__email__ = "acalvopic@unex.es"
__status__ = "Development"

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Union

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - tkinter may be unavailable on headless environments
    tk = None  # type: ignore
    ttk = None  # type: ignore

from PIL import Image, ImageTk


ImageLike = Union[Image.Image, "cv2.typing.MatLike"]  # type: ignore[name-defined]


@dataclass
class VisualizerConfig:
    title: str = "Generación DNI sintético"
    delay_seconds: float = 0.6
    max_width: int = 900
    max_height: int = 600


class DniVisualizer:
    """Visualizador sencillo basado en Tkinter para mostrar el proceso paso a paso."""

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        self.config = config or VisualizerConfig()
        self.enabled = tk is not None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_section: Optional[str] = None

        if not self.enabled:
            return

        try:
            self.root = tk.Tk()
        except Exception:
            self.enabled = False
            return

        self.root.title(self.config.title)
        self.root.geometry("980x720")
        self.root.minsize(640, 480)

        self.step_var = tk.StringVar(value="Esperando...")

        container = ttk.Frame(self.root, padding=12)
        container.pack(expand=True, fill="both")

        header = ttk.Frame(container)
        header.pack(fill="x")

        ttk.Label(header, textvariable=self.step_var, font=("TkDefaultFont", 14, "bold")).pack(
            side="left"
        )

        self.image_label = ttk.Label(container)
        self.image_label.pack(fill="both", expand=True, pady=(10, 10))

        log_frame = ttk.Frame(container)
        log_frame.pack(fill="both", expand=False)

        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_widget = tk.Text(
            log_frame,
            height=8,
            wrap="word",
            yscrollcommand=scrollbar.set,
            state="disabled",
        )
        self.log_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log_widget.yview)

        self.root.update_idletasks()

    def available(self) -> bool:
        return self.enabled

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.root.destroy()
        finally:
            self.enabled = False

    def section(self, name: str) -> None:
        if not self.enabled:
            return
        self._last_section = name
        self.log(f"=== {name} ===")

    def log(self, message: str) -> None:
        print(message)
        if not self.enabled:
            return
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")
        self._refresh()

    def show_step(
        self,
        description: str,
        image: ImageLike,
        note: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        pil_image = self._ensure_pil(image)
        pil_image = self._scale_image(pil_image)

        self.step_var.set(description)
        self._photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self._photo)

        if note:
            self.log(note)

        self._refresh()
        if self.config.delay_seconds > 0:
            time.sleep(self.config.delay_seconds)

    def _refresh(self) -> None:
        if not self.enabled:
            return
        self.root.update_idletasks()
        self.root.update()

    def _scale_image(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_w, max_h = self.config.max_width, self.config.max_height
        ratio = min(max_w / w, max_h / h, 1.0)
        if ratio < 1.0:
            new_size = (int(w * ratio), int(h * ratio))
            return image.resize(new_size, Image.LANCZOS)
        return image

    def _ensure_pil(self, image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy()

        # Asumimos array de OpenCV (BGR o BGRA)
        import numpy as np
        import cv2

        arr = image
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"No se puede convertir a imagen PIL: {type(image)!r}")

        if arr.ndim == 2:
            mode = "L"
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            mode = "RGB"
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            mode = "RGBA"
        else:
            raise ValueError(f"Formato de imagen no soportado: {arr.shape}")
        return Image.fromarray(arr, mode=mode)


__all__ = ["DniVisualizer", "VisualizerConfig"]

