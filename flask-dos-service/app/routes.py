from flask import Flask, jsonify
import numpy as np
from HIF_predict import HybridIsolationForest, train_with_synthetic_data, simulate_dos_attacks

app = Flask(__name__)

# Initialize the Hybrid Isolation Forest model
hif_model = HybridIsolationForest(t=10, psi=64)
train_with_synthetic_data(hif_model, n_samples=300)

@app.route('/simulate_dos', methods=['POST'])
def simulate_dos():
    n_samples = 10
    dos_scores = simulate_dos_attacks(hif_model, n_samples)
    
    # Umbral para decidir ataque
    threshold = 0.2335994392273122
    y_pred = dos_scores > threshold
    
    alerts = []
    for i, score in enumerate(dos_scores):
        status = "ATAQUE DETECTADO" if y_pred[i] else "Tráfico normal"
        alerts.append({"instance": i + 1, "score": score, "status": status})
    
    return jsonify(alerts)

if __name__ == '__main__':
    app.run(debug=True)