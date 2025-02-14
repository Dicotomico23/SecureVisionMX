from __future__ import absolute_import, print_function
import os
import glob
from PIL import Image
import utils
import torch
from torchvision import transforms
import numpy as np
from options.testing_options import TestOptions
import time
import json
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from socketio import Client

# Crear el objeto SocketIO
sio = Client()
# Solo se conecta si se ejecuta script_testing.py directamente
if __name__ == '__main__':
    try:
        sio.connect('http://127.0.0.1:8000')
    except Exception as e:
        print("No se pudo conectar a SocketIO:", e)

# Parsing options
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# Set data paths using DataRoot
data_root = opt.DataRoot  # Ejemplo: .../dataset/testing/IPAD
frames_dir = os.path.join(data_root, 'Test')
labels_dir = os.path.join(data_root, 'Test_gt')

print(f"Testing frames directory: {frames_dir}")
print(f"Testing labels directory: {labels_dir}")

# Model settings
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

# Load trained model
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

# Clase para inferencia en vivo y detección de anomalías
class AnomalyDetector:
    def __init__(self, model, device, frame_trans, img_crop_size, threshold=0.01):
        """
        threshold: umbral para determinar anomalías (puede ajustarse según la aplicación)
        """
        self.model = model
        self.device = device
        self.frame_trans = frame_trans
        self.img_crop_size = img_crop_size
        self.threshold = threshold
        self.alerts = []  # Lista de alertas detectadas

    def process_frame(self, img_path, frame_index):
        """
        Procesa una imagen para calcular el error y, en caso de anomalía,
        genera (o actualiza) una alerta.
        """
        from PIL import Image
        import time
        import numpy as np
        img = Image.open(img_path).convert('RGB')
        frame = self.frame_trans(img)                  # [3, H, W]
        frame = frame.unsqueeze(0).to(self.device)       # [1, 3, H, W]
        frame = frame.unsqueeze(2)                       # [1, 3, 1, H, W]

        with torch.no_grad():
            if opt.ModelName == 'AE':
                recon_frames = self.model(frame)
                recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
                input_np = utils.vframes2imgs(unorm_trans(frame.data), step=1, batch_idx=0)
                r = utils.crop_image(recon_np, self.img_crop_size) - utils.crop_image(input_np, self.img_crop_size)
                recon_error = np.mean(r ** 2)
            elif opt.ModelName == 'MemAE':
                recon_res = self.model(frame)
                recon_frames = recon_res['output']
                r = recon_frames - frame
                r = utils.crop_image(r, self.img_crop_size)
                sp_error_map = torch.sum(r ** 2, dim=1) ** 0.5
                s = sp_error_map.size()
                sp_error_vec = sp_error_map.view(s[0], -1)
                recon_error = torch.mean(sp_error_vec, dim=-1).cpu().item()
            else:
                print("Wrong ModelName.")
                recon_error = None

        # Si el error supera el umbral se considera anomalía
        if recon_error is not None and recon_error > self.threshold:
            # Si no hay alerta previa o la última alerta ya fue emitida, se crea una nueva
            if not self.alerts or self.alerts[-1].get("alert_sent", False):
                alert = {
                    "message": f"Anomalia detectada: el producto está en posición incorrecta (frame {frame_index} a {frame_index})",
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "alert_sent": False
                }
                self.alerts.append(alert)
            else:
                # Actualiza el último alert para reflejar el rango de frames
                alert = self.alerts[-1]
                alert["last_frame"] = frame_index
                alert["message"] = f"Anomalia detectada: el producto está en posición incorrecta (frame {alert['first_frame']} a {alert['last_frame']})"
                # Emitir alerta solo la primera vez
                if sio.connected:
                    sio.emit('alert', alert)
                    alert["alert_sent"] = True
                else:
                    print("SocketIO no está conectado; no se envió la alerta.")
                # Actualizamos el rango sin reemitir la alerta
                self.alerts[-1]["last_frame"] = frame_index
            alerta = self.alerts[-1]
            print(f"ALERTA: {alerta['message']} desde frame {alerta['first_frame']} hasta {alerta['last_frame']} con error {recon_error}")

        return recon_error

    def run_live_inference(self, image_list, sleep_time=0.1):
        """
        Simula una transmisión en vivo procesando cada imagen secuencialmente,
        mostrando los resultados y guardando alertas cuando se detecta anomalía.
        sleep_time: tiempo de espera entre frames para simular video en vivo.
        """
        recon_error_list = []
        for idx, img_path in enumerate(image_list):
            error = self.process_frame(img_path, idx+1)
            if error is not None:
                recon_error_list.append(error)
            print(f"Procesada imagen {idx+1}/{len(image_list)}")
            time.sleep(sleep_time)
        return recon_error_list

    def save_alerts(self, json_filepath):
        """
        Guarda todas las alertas en un archivo JSON.
        """
        with open(json_filepath, "w") as json_file:
            json.dump(self.alerts, json_file, indent=4)
        print(f"Alertas guardadas en {json_filepath}")

# ----------------------------
# Código de prueba del módulo en este mismo script:
# Obtener la lista de imágenes y usar solo las primeras 100 (primer video)
image_list = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
image_list = image_list[:100]
num_images = len(image_list)
print(f"Se encontraron {num_images} frames en {frames_dir}")

# Crear instancia del detector de anomalías con un umbral (se puede ajustar)
detector = AnomalyDetector(model, device, frame_trans, img_crop_size, threshold=0.01)

# Ejecutar inferencia en vivo simulada (esto se ejecuta al importar script_testing.py solo si __name__ == '__main__')
if __name__ == '__main__':
    recon_error_list = detector.run_live_inference(image_list, sleep_time=0.05)
    np.save(os.path.join(te_res_path, "recon_errors.npy"), recon_error_list)
    alerts_json_path = os.path.join(te_res_path, "alerts.json")
    detector.save_alerts(alerts_json_path)
    utils.eval_video(data_root, te_res_path, is_show=False)