# myapp/settings.py

import os
from django.conf import settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIC_URL = '/static/'  # Cambiado a '/static/' para que coincida con la configuración estándar
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'static', 'myapp')  # Cambiado a 'static_root/myapp'
