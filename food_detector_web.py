# food_detector_web.py

from __future__ import annotations
import base64
import os
import random
import time
import numpy as np
from io import BytesIO
from typing import Any, Dict, List
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
# Importe o detector. Garanta que o ficheiro detector.py está na mesma pasta.
from detector import AdvancedFoodDetector

# --- CONFIGURAÇÕES ---
MODEL_PATH, NAMES_PATH = "modelo_food_advanced_final.h5", "class_names.txt"
# !!! ATENÇÃO: Verifique se este caminho está 100% correto para o seu sistema !!!
DATA_DIR = "/Users/renzotognella/Github/Trabalho_PDI/bounding_box"
UPLOAD_FOLDER, MAX_CONTENT_BYTES = "uploads", 16 * 1024 * 1024

# --- INICIALIZAÇÃO DA APLICAÇÃO ---
app = Flask(__name__)
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, MAX_CONTENT_LENGTH=MAX_CONTENT_BYTES)

try:
    afd = AdvancedFoodDetector(MODEL_PATH, NAMES_PATH)
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o modelo de deteção em '{MODEL_PATH}'. Verifique o caminho.")
    print(f"Erro do sistema: {e}")
    afd = None

available_classes: List[Dict[str, Any]] = []

# --- FUNÇÕES AUXILIARES ---

def create_upload_folder():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def image_to_base64_from_path(path: str, max_xy: int = 800) -> str | None:
    """Converte uma imagem de um ficheiro para uma string Base64 de forma segura."""
    if not os.path.exists(path):
        print(f"Erro: Ficheiro não encontrado em {path}")
        return None
    try:
        with Image.open(path) as img:
            img.thumbnail((max_xy, max_xy), Image.Resampling.LANCZOS)
            buf = BytesIO()
            # Converte imagens com transparência (PNG) para RGB antes de salvar como JPEG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Erro ao converter imagem de {path} para Base64: {e}")
        return None

def pil_image_to_base64(img: Image.Image, max_xy: int = 800) -> str | None:
    """Converte um objeto de imagem PIL para uma string Base64."""
    try:
        img.thumbnail((max_xy, max_xy), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Erro ao converter imagem PIL para Base64: {e}")
        return None

def numpy_to_native(obj: Any) -> Any:
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, dict): return {k: numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [numpy_to_native(i) for i in obj]
    return obj

def load_available_classes():
    global available_classes
    available_classes = []
    if not os.path.isdir(DATA_DIR):
        print(f"AVISO CRÍTICO: O diretório de dados '{DATA_DIR}' não foi encontrado. A seleção de exemplos não funcionará.")
        return
    for name in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, name)
        if not os.path.isdir(p): continue
        imgs = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            available_classes.append({"name": name, "path": p, "images": imgs, "count": len(imgs)})

# --- ROTAS DA APLICAÇÃO ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/classes")
def get_classes():
    return jsonify(available_classes)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"success": False, "error": "Nenhum ficheiro enviado"}), 400
    
    filename = secure_filename(f.filename)
    unique_filename = f"uploaded_{int(time.time())}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        f.save(path)
        return jsonify({"success": True, "filepath": path})
    except Exception as e:
        print(f"Erro ao salvar o ficheiro {path}: {e}")
        return jsonify({"success": False, "error": "Não foi possível salvar o ficheiro no servidor."}), 500

@app.route("/api/select_from_class", methods=["POST"])
def select_from_class():
    data = request.get_json(silent=True) or {}
    class_name = data.get("class_name")
    cls = next((c for c in available_classes if c["name"] == class_name), None)
    
    if not cls:
        return jsonify({"success": False, "error": f"Classe '{class_name}' não encontrada."}), 404
    
    if not cls["images"]:
        return jsonify({"success": False, "error": f"Classe '{class_name}' não contém imagens."}), 404
            
    img_name = random.choice(cls["images"])
    path = os.path.join(cls["path"], img_name)
    
    image_data = image_to_base64_from_path(path)
    if not image_data:
        error_msg = f"Ficheiro de imagem não encontrado ou erro de leitura. Verifique o caminho em DATA_DIR. Caminho esperado: {path}"
        print(f"AVISO: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 404

    return jsonify({
        "success": True, 
        "filepath": path,
        "image_data": image_data
    })

@app.route("/api/detect", methods=["POST"])
def run_detection():
    if afd is None:
        return jsonify({"success": False, "error": "O modelo do detector não foi carregado."}), 500

    img_path = (request.get_json(silent=True) or {}).get("image_path")
    if not img_path or not os.path.exists(img_path):
        return jsonify({"success": False, "error": f"Ficheiro de imagem não encontrado no servidor no caminho: {img_path}"}), 404

    try:
        dets, processed_image_pil = afd.run(img_path)
        native_dets = numpy_to_native(dets)
        result_image_data = pil_image_to_base64(processed_image_pil)

        if result_image_data is None:
            return jsonify({"success": False, "error": "Falha ao processar a imagem resultante."}), 500

        return jsonify({
            "success": True,
            "detected_foods": native_dets,
            "result_image_data": result_image_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Ocorreu um erro interno durante a deteção: {e}"}), 500

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    create_upload_folder()
    load_available_classes()
    app.run(debug=True, host="0.0.0.0", port=8080)
