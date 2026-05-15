import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ai_pipeline import AIProcessor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inisialisasi engine AI
processor = AIProcessor()

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file gambar"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File kosong"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Eksekusi AI Pipeline
            hasil = processor.analyze(filepath)
            
            # Hapus gambar sementara
            os.remove(filepath)
            
            return jsonify({"status": "sukses", "data": hasil}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)