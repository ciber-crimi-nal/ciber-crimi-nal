# Pipeline de Generación de DNIs con IA

## Descripción

Este pipeline modificado genera DNIs sintéticos siguiendo un nuevo flujo:

1. **Genera datos aleatorios** (edad, género, nombres, etc.)
2. **Crea imagen con IA** usando esos datos
3. **Compone el DNI** con la imagen generada

## Diferencias con el pipeline original

### Pipeline Original
```
Foto existente → Detectar edad/género → Generar datos DNI → Componer DNI
```

### Nuevo Pipeline
```
Generar datos DNI → Generar foto con IA → Componer DNI
```

## Archivos modificados/nuevos

- **`dni_pipeline.py`**: Pipeline modificado para usar generación IA
- **`main.py`**: Script principal que usa el nuevo pipeline
- **`ImageGeneration.py`**: Ya existente, genera imágenes con IA

## Requisitos

1. Tener el servidor de generación de imágenes corriendo en `http://127.0.0.1:8888`
2. Las dependencias del proyecto original:
   - opencv-python
   - pandas
   - Pillow
   - rembg
   - etc.

Instalar y revisar `requirements.txt`

## Uso

### Generación básica (edad y género aleatorios)

```bash
python main.py
```

Esto genera:
- `salidas/dni_anverso.png`
- `salidas/dni_reverso.png`
- `salidas/dni_campos.json`

### Especificar edad y género

```bash
python main.py --age 35 --gender male
```

### Generar múltiples DNIs

```bash
python main.py --count 10
```

Genera: `dni_000_anverso.png`, `dni_001_anverso.png`, etc.

### Mantener las imágenes generadas por IA

Por defecto, las imágenes de IA se eliminan después de crear el DNI. Para mantenerlas:

```bash
python main.py --keep-ai-images
```

### Visualización paso a paso

```bash
python main.py --visualize
```

### Opciones avanzadas

```bash
python main.py \
  --age 28 \
  --gender female \
  --count 5 \
  --output-dir mis_dnis \
  --label-prefix persona \
  --keep-ai-images \
  --visualize \
  --visualize-delay 1.0
```

## Estructura de salida

Cada DNI genera estos archivos:

```
salidas/
├── dni_anverso.png          # Anverso del DNI
├── dni_reverso.png          # Reverso del DNI
├── dni_campos.json          # Metadatos (incluye person_data)
└── dni_ai_face.png          # Imagen generada (si --keep-ai-images)
```

### Contenido del JSON

```json
{
  "apellido1": "GARCIA",
  "apellido2": "LOPEZ",
  "nombre": "MARIA",
  "sexo": "F",
  "nacionalidad": "ESP",
  "fecha de nacimiento": "15 03 1988",
  "dni": "12345678A",
  ...
  "ai_generated": true,
  "person_data": {
    "age": 35,
    "gender": "female"
  }
}
```

## Ventajas del nuevo pipeline

1. **No necesita dataset de fotos no reales**: Genera todo sintéticamente
2. **Control total sobre características**: Puedes especificar edad y género exactos
3. **Consistencia**: La foto generada coincide con los datos del DNI
4. **Reproducibilidad**: Los datos se generan primero, permitiendo regenerar la imagen si es necesario

## Limitaciones

1. **Requiere el modelo de generación de imágenes**: Debe estar corriendo para la generación de imágenes
2. **Tiempo de generación**: Más lento que usar fotos existentes (depende del modelo)
3. **Calidad variable**: La calidad depende tanto del modelo usado como de prompting.

## Ejemplo completo de ejecución

```bash
# 1. Asegúrate de que el servidor del modelo de generación de imágenes está corriendo

# 2. Genera un DNI con visualización
python main.py \
  --age 42 \
  --gender male \
  --visualize \
  --keep-ai-images \
  --output-dir mi_dni

# 3. Revisa los resultados
ls mi_dni/
# dni_anverso.png  dni_reverso.png  dni_campos.json  dni_ai_face.png
```

## Ejecución del modelo en local o remoto

Con el siguiente docker compose se puede levantar el modelo donde se disponga de suficiente computo.

```yml
version: "3.9"

services:
  fooocus-api:
    image: konieshadow/fooocus-api
    container_name: fooocus-api
    ports:
      - "8888:8888"
    environment:
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      NVIDIA_VISIBLE_DEVICES: all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    restart: unless-stopped
```

Las pruebas de "rendimiento" realizadas se han ejecutado con las siguientes specs:

| Componente | Modelo | Detalles / Capacidad | Versión del Driver |
| :--- | :--- | :--- | :--- |
| **Procesador (CPU)** | Intel Core **i7-14700HX** | **20 Núcleos** (Physical) / 28 Hilos | - |
| **Memoria RAM** | Sistema Total | **32 GB** | - |
| **GPU Dedicada** | **NVIDIA GeForce RTX 4060** | **8 GB VRAM** (8188 MiB)* | 32.0.15.7703 |
| **GPU Integrada** | Intel UHD Graphics | 2 GB | 32.0.101.5972 |


* **CUDA Cores:** 3072
* **Tipo de Memoria:** GDDR6
* **Bus:** 128-bit

## Comparación de rendimiento

| Método | Tiempo por DNI | Ventajas | Desventajas |
|--------|---------------|----------|-------------|
| **Original** (fotos no reales no tratadas) | ~2-3s | Rápido, fotos no tratadas | Necesita dataset |
| **Nuevo** (generación IA) | ~10-30s | No necesita dataset, controlable | Más lento, necesita el modelo |

## Próximas mejoras

- [ ] Cachear imágenes generadas por parámetros similares
- [ ] Permitir especificar más características (etnia, estilo de pelo, etc.)
- [ ] Generación en batch más eficiente
- [ ] Fallback al pipeline original si falla la IA