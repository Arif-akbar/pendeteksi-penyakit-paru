import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ai_pipeline import AIProcessor
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".dcm", ".dicom"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
UPLOAD_DIR.mkdir(exist_ok=True)

# Inisialisasi engine AI
processor = AIProcessor()

@app.route('/', methods=['GET'])
def index():
    if (FRONTEND_DIR / "index.html").exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")

    return jsonify({
        "status": "online",
        "pesan": "Lung AI API Engine Aktif. Silakan tembak endpoint /api/predict dengan metode POST."
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "endpoint": "/api/predict",
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file gambar"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File kosong"}), 400

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Format file belum didukung",
            "supported": sorted(ALLOWED_EXTENSIONS)
        }), 400

    filepath = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
    file.save(filepath)

    try:
        hasil = processor.analyze(str(filepath))
        return jsonify({"status": "sukses", "data": hasil}), 200
    except Exception as e:
        app.logger.exception("Prediksi gagal")
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath.exists():
            filepath.unlink()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
