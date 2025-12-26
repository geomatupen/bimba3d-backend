from pathlib import Path
from typing import List

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "projects"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# API
API_PORT = 8005

# CORS - Update these for production
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:5174",  # Alternate Vite dev port
    "http://localhost:3000",  # Alternative frontend port
    "http://localhost:8080",  # Another common port
]

# File upload settings
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
