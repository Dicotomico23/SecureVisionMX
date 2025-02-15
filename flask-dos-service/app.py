from flask import Flask, render_template, jsonify, request
import sys
import numpy as np
from HIF import HybridIsolationForest
from flask_cors import CORS
import os
import glob
from PIL import Image
import utils
import torch
from torchvision import transforms
import time
import json
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from options.testing_options import TestOptions
from script_testing import AnomalyDetector  
from flask_socketio import SocketIO
from AI_Agent import agent_executor, generate_pdf_report  # 🔹 Importar el agente y la función de reportes

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# ===============================
# CARGA DE CONFIGURACIÓN Y MODELO
# ===============================
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=False)
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

data_root = opt.DataRoot  # Ejemplo: .../dataset/testing/IPAD
frames_dir = os.path.join(data_root, 'Test')
labels_dir = os.path.join(data_root, 'Test_gt')

batch_size_in = opt.BatchSize
chnum_in_ = opt.ImgChnNum
framenum_in_ = opt.FrameNum
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres
img_crop_size = 0

model_setting = utils.get_model_setting(opt)

# Model path
model_root = opt.ModelRoot
model_path = opt.ModelFilePath if opt.ModelFilePath else os.path.join(model_root, model_setting + '.pt')

# Test result path
te_res_root = opt.OutRoot
te_res_path = os.path.join(te_res_root, 'res_' + model_setting)
utils.mkdir(te_res_path)

if opt.ModelName == 'AE':
    model = AutoEncoderCov3D(chnum_in_)
elif opt.ModelName == 'MemAE':
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong ModelName.')

model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

frame_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * chnum_in_, [0.5] * chnum_in_)
])
unorm_trans = utils.UnNormalize(mean=[0.5] * chnum_in_, std=[0.5] * chnum_in_)

# Crear instancia del detector de anomalías
detector = AnomalyDetector(model, device, frame_trans, img_crop_size, threshold=0.01)

# ===============================
# DEFINICIÓN DE ENDPOINTS DE LA API
# ===============================

@app.route('/api/infer', methods=['GET'])
def infer():
    """ Ejecuta el detector sobre imágenes y devuelve los resultados. """
    image_list = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))[:10]
    start_time = time.time()
    errors = detector.run_live_inference(image_list, sleep_time=0)
    end_time = time.time()
    
    detector.save_alerts(os.path.join(te_res_path, "alerts.json"))
    
    return jsonify({
        'inference_time': end_time - start_time,
        'errors': errors,
        'alerts': detector.alerts
    })

# Inicializar el modelo de detección de anomalías en la red
hif = HybridIsolationForest(t=10, psi=64)

def train_with_synthetic_data(hif_model, n_samples=300):
    """ Entrena el modelo HIF con datos sintéticos. """
    X_train = np.random.rand(n_samples, 78) * 0.5
    y_train = np.zeros(n_samples)
    hif_model.fit(X_train, y_train)

train_with_synthetic_data(hif, n_samples=300)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate_dos_attacks', methods=['POST'])
def simulate_dos_attacks():
    """ Simula ataques DoS y retorna alertas si se detectan. """
    X_dos = np.random.rand(10, 78) * 0.8 + 0.2
    for row in X_dos:
        row[np.random.choice(X_dos.shape[1], size=10, replace=False)] *= 1.2

    dos_scores = hif.predict(X_dos)
    threshold = 0.2335994392273122
    alerts = [f"[DOS] Instancia {i+1}: score={score:.4f} -> ATAQUE DETECTADO"
              for i, score in enumerate(dos_scores) if score > threshold]

    return jsonify(alerts)

@app.route('/api/door-control', methods=['POST'])
def toggle_door():
    """ Control de puertas (ejemplo de automatización). """
    return jsonify({'door_state': 'Abierto'})

# ===============================
# INTEGRACIÓN DEL AGENTE INTELIGENTE
# ===============================

@app.route('/api/ai-agent', methods=['POST'])
def ai_agent():
    """
    Endpoint para interactuar con el agente inteligente.
    Recibe una consulta y devuelve la respuesta generada.
    """
    data = request.get_json()
    user_input = data.get("query", "Consulta de prueba")

    # Ejecutar el agente
    results = agent_executor.run(user_input)

    # Generar reporte si es necesario
    if "reporte" in user_input.lower():
        report_path = generate_pdf_report(results)
        return jsonify({'response': "Reporte generado con éxito.", 'report_path': report_path})

    return jsonify({'response': results})

@app.route('/api/agent-query', methods=['POST'])
def agent_query():
    """
    Endpoint para interactuar con el agente inteligente.
    Recibe una consulta y devuelve la respuesta generada.
    """
    data = request.get_json()
    user_input = data.get("query", "Consulta de prueba")

    # Ejecutar el agente
    results = agent_executor.invoke(user_input)

    # Extraer la respuesta; revisar por 'response' o 'output'
    output = None
    if isinstance(results, dict):
        output = results.get("response") or results.get("output")
    else:
        output = results

    if not output:
        output = "ChatOllama no conectado"

    print("Consulta:", user_input)
    print("Respuesta:", output)

    return jsonify({'response': output})

# ===============================
# MANEJO DE SOCKETIO PARA CHAT EN TIEMPO REAL
# ===============================
@socketio.on('user_message')
def handle_user_message(data):
    user_input = data['message']
    response = agent_executor.run(user_input)
    socketio.emit('bot_response', {'message': response})

# ===============================
# EJECUCIÓN DEL SERVIDOR
# ===============================
if __name__ == '__main__':
    print("Sesión iniciada: el servidor está listo para realizar inferencias con los modelos almacenados y el agente inteligente.")
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)