from PIL import Image
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile

def convertir_a_escala_de_grises(imagen):
    # Abrir la imagen
    img = Image.open(imagen)

    # Convertir a escala de grises
    img_gris = img.convert("L")

    # Guardar la imagen en memoria
    buffer = BytesIO()
    img_gris.save(buffer, format="PNG")

    # Crear un archivo de imagen en memoria
    imagen_gris = InMemoryUploadedFile(
        buffer, None, 'imagen_gris.png', 'image/png', buffer.tell(), None
    )

    return imagen_gris