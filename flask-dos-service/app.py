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
import numpy as np
import time
import json
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from options.testing_options import TestOptions
from script_testing import AnomalyDetector  
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})
socketio = SocketIO(app, cors_allowed_origins="http://127.0.0.1:5500")


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
if opt.ModelFilePath:
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root, model_setting + '.pt')

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

if chnum_in_ == 1:
    norm_mean = [0.5]
    norm_std = [0.5]
elif chnum_in_ == 3:
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

frame_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# Crear instancia del detector de anomalías
detector = AnomalyDetector(model, device, frame_trans, img_crop_size, threshold=0.01)

# ===============================
# DEFINICIÓN DE ENDPOINTS DE LA API
# ===============================
@app.route('/api/infer', methods=['GET'])
def infer():
    """
    Ejecuta el detector sobre las primeras 100 imágenes (simulando video en vivo),
    guarda alertas en un archivo JSON y retorna los resultados.
    """
    image_list = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))[:10]
    start_time = time.time()
    errors = detector.run_live_inference(image_list, sleep_time=0)
    end_time = time.time()
    inference_time = end_time - start_time
    alerts_json_path = os.path.join(te_res_path, "alerts.json")
    detector.save_alerts(alerts_json_path)
    return jsonify({
        'inference_time': inference_time,
        'errors': errors,
        'alerts': detector.alerts
    })

# Initialize the model
hif = HybridIsolationForest(t=10, psi=64)

# Train with synthetic data
def train_with_synthetic_data(hif_model, n_samples=300):
    X_train = np.random.rand(n_samples, 78) * 0.5
    y_train = np.zeros(n_samples)
    hif_model.fit(X_train, y_train)

train_with_synthetic_data(hif, n_samples=300)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate_dos_attacks', methods=['POST'])
def simulate_dos_attacks():
    X_dos = np.random.rand(10, 78) * 0.8 + 0.2
    for row in X_dos:
        idxs = np.random.choice(X_dos.shape[1], size=10, replace=False)
        row[idxs] *= 1.2

    dos_scores = hif.predict(X_dos)
    threshold = 0.2335994392273122
    y_pred = dos_scores > threshold

    alerts = []
    for i, score in enumerate(dos_scores):
        if y_pred[i]:
            alerts.append(f"[DOS] Instancia {i+1}: score={score:.4f} -> ATAQUE DETECTADO")

    return jsonify(alerts)

@app.route('/api/door-control', methods=['POST'])
def toggle_door():
    # Ejemplo de endpoint para el control de la puerta
    return jsonify({'door_state': 'Abierto'})

# ===============================
# EJECUCIÓN DEL SERVIDOR
# ===============================
if __name__ == '__main__':
    print("Sesión iniciada: el servidor está listo para realizar inferencias con los modelos almacenados.")
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)