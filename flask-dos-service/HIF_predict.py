import sys
sys.path.append("/Users/dilofac/Documents/Valtron CV/models/memae-anomaly-detection/")
from HIF import HybridIsolationForest
import numpy as np

def train_with_synthetic_data(hif_model, n_samples=300):
    """
    Para evitar hojas vacías, primero entrenamos el modelo con datos sintéticos
    (e.g. tráfico normal). De esa manera, las particiones tendrán información y
    no habrá mean of empty slice.
    """
    # Genera datos aleatorios normalizados (puedes ajustar la dimensión si tu modelo usa más/menos features)
    X_train = np.random.rand(n_samples, 78) * 0.5
    # Etiquetas de 0 para indicar normal
    y_train = np.zeros(n_samples)
    hif_model.fit(X_train, y_train)

def simulate_dos_attacks(hif_model, n_samples=1000):
    """
    Crea datos sintéticos que simulan un ataque DOS y los evalúa usando
    la función 'predict' y demás utilidades de tu modelo (HybridIsolationForest).
    """
    # Generamos datos con valores altos para simular tráfico sospechoso
    X_dos = np.random.rand(n_samples, 78) * 0.8 + 0.2  # desplaza valores a la banda [0.2, 1.0]

    # Intensificamos columnas al azar para simular agresión
    for row in X_dos:
        idxs = np.random.choice(X_dos.shape[1], size=10, replace=False)
        row[idxs] *= 1.2

    # Agregamos un retardo de 3 segundos antes de iniciar la detección de anomalías en red
    time.sleep(3)

    # Obtenemos puntuaciones de anomalía usando la función 'predict'
    dos_scores = hif_model.predict(X_dos)

    # Umbral para decidir ataque
    threshold = 0.2335994392273122
    y_pred = dos_scores > threshold

    # Contar alertas cada 10 muestras
    alert_count = 0
    for i in range(0, n_samples, 10):
        positive_alerts = np.sum(y_pred[i:i+10])
        negative_alerts = 10 - positive_alerts
        if negative_alerts > positive_alerts:
            alert_count += 1
            print(f"[DOS] Grupo {i//10 + 1}: Anomalía detectada")

    print(f"Total de grupos con anomalías: {alert_count}")

if __name__ == "__main__":
    # Ejemplo de uso:
    hif = HybridIsolationForest(t=10, psi=64)
    # Entrenamos con datos sintéticos para evitar leaves vacías
    train_with_synthetic_data(hif, n_samples=300)

    # Ahora simulamos el ataque DOS
    simulate_dos_attacks(hif, n_samples=1000)