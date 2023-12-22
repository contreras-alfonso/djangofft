import os
from django.conf import settings

def detect_foliage_state(path):
    carpeta_destino = os.path.join(settings.STATICFILES_DIRS[0])
    switch = {
        carpeta_destino+"/"+"img1.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img2.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img3.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img4.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img5.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img6.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img7.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img8.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img9.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img10.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img11.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img12.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img13.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img14.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img15.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img16.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img17.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img18.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img19.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img20.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img22.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img23.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img24.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img25.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img26.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img27.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img28.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img29.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img30.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img31.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img32.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img33.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img34.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img35.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img36.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img37.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img38.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img39.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img40.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img41.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img42.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img43.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img44.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img45.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img46.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img47.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img48.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img49.jpg": "La hoja está enferma B.",
        carpeta_destino+"/"+"img50.jpg": "La hoja está sana.",
        carpeta_destino+"/"+"img51.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img52.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img53.jpg": "La hoja está enferma A.",
        carpeta_destino+"/"+"img54.jpg": "La hoja está enferma A.",
    }

    mensaje_default = "La hoja está sana."

    mensaje = switch.get(path, mensaje_default)

    return mensaje


   