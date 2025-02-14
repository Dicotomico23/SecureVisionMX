import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from joblib import Parallel, delayed
import multiprocessing

class exNode:
    #def __init__(self, S, SLab, c_value):
    def __init__(self, c_value, CS):
        #self.S = S
        #self.SLab = SLab
        #self.CS = np.mean(S, axis=0)
        self.CS = CS
        self.c_lenS = c_value
        self.Xa = []
        self.labels = []
        self.Ca = None

class inNode:
    def __init__(self, left, right, splitDim, splitVal):
        self.left = left
        self.right = right
        self.splitDim = splitDim
        self.splitVal = splitVal

class HybridIsolationForest:
    def __init__(self, t=10, psi=64):
        self.t = t
        self.psi = psi
        self.lmax = math.ceil(math.log2(psi))
        self.forest = []
        self.alpha1 = 0.08375522683976655
        self.alpha2 = 0.9987820501864242

    def c(self, size):
        if size <= 1:
            return 0
        return 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size


    def hiTree(self, S, SLab, l, randSeed=None):

        if randSeed is not None:
          np.random.seed(randSeed)

        if l >= self.lmax or len(S) <= 1:
            c_value = self.c(len(S))
            return exNode(S, SLab, c_value)
        else:
            q = np.random.randint(0, S.shape[1])
            p = np.random.uniform(np.min(S[:, q]), np.max(S[:, q]))

            mask_left = S[:, q] < p
            mask_right = ~mask_left

            Sl = S[mask_left]
            Sr = S[mask_right]
            SLabl = SLab[mask_left]
            SLabr = SLab[mask_right]

            left = self.hiTree(Sl, SLabl, l + 1)
            right = self.hiTree(Sr, SLabr, l + 1)
            return inNode(left, right, q, p)

    def hiTree_batch(self, X_train, l, randSeed=None):

        if l == 0:
          if randSeed is not None:
            np.random.seed(randSeed)

          subset_indices = np.random.choice(X_train.shape[0], self.psi, replace=False)

          S = X_train[subset_indices]
          SLab = np.zeros(self.psi)

        else:
          S = X_train
          SLab = np.zeros(X_train.shape[0])

        if l >= self.lmax or len(S) <= 1:
            c_value = self.c(self.psi)
            centroid = None
            if len(S) > 0:
                centroid = np.mean(S, axis=0)
            print(f'Centroid: {centroid}')
            print(f'c_value: {c_value}')
            print(f'len(S): {len(S)}')
            return exNode(c_value, centroid)

        else:

            q = np.random.randint(0, X_train.shape[1])
            p = np.random.uniform(np.min(S[:, q]), np.max(S[:, q]))

            mask_left = S[:, q] < p
            Sl = S[mask_left]
            Sr = S[~mask_left]
            #SLabl = SLab[mask_left]
            #SLabr = SLab[~mask_left]

            left = self.hiTree_batch(Sl, l + 1, randSeed)
            right = self.hiTree_batch(Sr, l + 1, randSeed)
            return inNode(left, right, q, p)

    def hiTree_iterative(self, S, SLab, randSeed=None):

        if randSeed is not None:
          np.random.seed(randSeed)

        root = None
        stack = []
        stack.append((S, SLab, 0, None, None))

        internal_nodes = []
        leaf_nodes = []

        while stack:
            S_sub, SLab_sub, depth, parent_node, is_left = stack.pop()

            if depth >= self.lmax or len(S_sub) <= 1:
                c_value = self.c(len(S_sub))
                node = exNode(c_value=c_value, CS = 0)

                leaf_nodes.append(node)

            else:
                q = np.random.randint(0, S_sub.shape[1])
                p = np.random.uniform(np.min(S_sub[:, q]), np.max(S_sub[:, q]))

                mask_left = S_sub[:, q] < p
                Sl, SLabl = S_sub[mask_left], SLab_sub[mask_left]
                Sr, SLabr = S_sub[~mask_left], SLab_sub[~mask_left]

                node = inNode(left=None, right=None, splitDim=q, splitVal=p)

                internal_nodes.append(node)

                stack.append((Sr, SLabr, depth + 1, node, False))
                stack.append((Sl, SLabl, depth + 1, node, True))

            if parent_node is not None:
                if is_left:
                    parent_node.left = node
                else:
                    parent_node.right = node
            else:
                root = node
        return root

    def generate_subsets(self, X):
        subsets = []
        n_samples = X.shape[0]

        for _ in range(self.t):
            if self.psi > n_samples:
                raise ValueError("psi cannot be larger than the number of samples in X")
            subset_indices = np.random.choice(n_samples, self.psi, replace=False)
            subsets.append(subset_indices)

        return subsets

    def generate_subset(self, X):
        n_samples = X.shape[0]
        subset_indices = np.random.choice(n_samples, self.psi, replace=False)
        return subset_indices


#______________________________________________________________________________________________________________
###############################################################################################################
#--------------------------------------------------------------------------------------------------------------

    def addAnomaly(self, x, xlab): #parallelize for every tree
        for tree in self.forest:
            self._addAnomaly(x, xlab, tree)

    def _addAnomaly(self, x, xlab, T):
        if isinstance(T, exNode):
            T.labels.append(xlab)
            T.Xa.append(x)
        else:
            a = T.splitDim
            if x[a] < T.splitVal:
                self._addAnomaly(x, xlab, T.left)
            else:
                self._addAnomaly(x, xlab, T.right)

    def computeAnomalyCentroid(self): #parallelize for every tree
        for tree in self.forest:
            self._computeAnomalyCentroid(tree)

    def _computeAnomalyCentroid(self, T):
        if isinstance(T, exNode):
            T.Ca = None
            if len(T.Xa) > 0:
                T.Ca = np.mean(T.Xa, axis=0)
        else:
            self._computeAnomalyCentroid(T.left)
            self._computeAnomalyCentroid(T.right)

    def euclidean_distance(self, x, y):
        return np.linalg.norm(x - y)

    def hiScore(self, x, T, e=0):
        if isinstance(T, exNode):
            h_x = e + self.c(10)
            delta_x = self.euclidean_distance(x, T.CS)
            delta_a_x = self.euclidean_distance(x, T.Ca) if T.Ca is not None else 0
            return h_x, delta_x, delta_a_x
        else:
            a = T.splitDim
            if x[a] < T.splitVal:
                return self.hiScore(x, T.left, e + 1)
            else:
                return self.hiScore(x, T.right, e + 1)

    def hiScore_iter(self, x, T):
        e = 0
        node = T
        while not isinstance(node, exNode):
            a = node.splitDim
            if x[a] < node.splitVal:
                node = node.left
            else:
                node = node.right
            e += 1
        # node es exNode aquí
        h_x = e + node.c_lenS  # sumamos e + c(len(S)) precomputado
        # Calculamos distancias
        delta_x = np.linalg.norm(x - node.CS) if node.CS is not None else 0.0
        if node.Ca is not None:
            delta_a_x = np.linalg.norm(x - node.Ca)
        else:
            delta_a_x = 0.0
        return h_x, delta_x, delta_a_x

    def compute_scores_batch(self, X):

        n = len(X)
        h_scores = np.zeros(n)
        delta_scores = np.zeros(n)
        delta_a_scores = np.zeros(n)

        for tree in self.forest:
            for i in range(n):
                h_x, dx, da = self.hiScore_iter(X[i], tree)
                h_scores[i] += h_x
                delta_scores[i] += dx
                delta_a_scores[i] += da

        h_scores /= len(self.forest)
        delta_scores /= len(self.forest)
        delta_a_scores /= len(self.forest)
        return h_scores, delta_scores, delta_a_scores

    def _compute_scores_one_tree(self, tree, X):

        n = len(X)
        h_scores = np.zeros(n)
        delta_scores = np.zeros(n)
        delta_a_scores = np.zeros(n)

        for i in range(n):
            h_x, dx, da = self.hiScore_iter(X[i], tree)
            h_scores[i] += h_x
            delta_scores[i] += dx
            delta_a_scores[i] += da

        return h_scores, delta_scores, delta_a_scores

    def compute_scores_batch_cpuParallelization(self, X):

        n_trees = len(self.forest)
        print(f"Starting parallel scoring for {n_trees} trees...")

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(self._compute_scores_one_tree)(tree, X)
            for tree in self.forest
        )

        n = len(X)
        h_scores = np.zeros(n)
        delta_scores = np.zeros(n)
        delta_a_scores = np.zeros(n)

        for (h_s, dx_s, da_s) in results:
            h_scores += h_s
            delta_scores += dx_s
            delta_a_scores += da_s

        h_scores /= n_trees
        delta_scores /= n_trees
        delta_a_scores /= n_trees

        return h_scores, delta_scores, delta_a_scores

    def compute_scores(self, x):
        h_scores = []
        delta_scores = []
        delta_a_scores = []

        for tree in self.forest:
            h_x, delta_x, delta_a_x = self.hiScore(x, tree)
            h_scores.append(h_x)
            delta_scores.append(delta_x)
            delta_a_scores.append(delta_a_x)

        return np.mean(h_scores), np.mean(delta_scores), np.mean(delta_a_scores)

    def normalize_scores(self, scores):
        scores = np.array(scores)
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def aggregate_scores_iter_batch(self, X, alpha1, alpha2):
        raw_s_scores = []
        raw_sc_scores = []
        raw_sa_scores = []

        for x in X:
            h_mean, delta_mean, delta_a_mean = self.compute_scores(x)
            raw_s_scores.append(h_mean)
            raw_sc_scores.append(delta_mean)
            raw_sa_scores.append(delta_mean / delta_a_mean if delta_a_mean != 0 else 0)

        s_scores = self.normalize_scores(raw_s_scores)
        sc_scores = self.normalize_scores(raw_sc_scores)
        sa_scores = self.normalize_scores(raw_sa_scores)

        shif_scores = alpha2 * (alpha1 * s_scores + (1 - alpha1) * sc_scores) + (1 - alpha2) * sa_scores
        return shif_scores

    def optimize_alpha(self, X, y):
        def objective(params):
            alpha1, alpha2 = params
            #print(f'Trying with alpha1: {alpha1}, alpha2: {alpha2}')
            scores = self.aggregate_scores(X, alpha1, alpha2)
            f1 = f1_score(y, scores > 0.5, average='macro')
            return -f1
        bounds = [(0, 1), (0, 1)]
        result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, popsize=15)
        self.alpha1, self.alpha2 = result.x
        print(f'Optimal alpha1: {self.alpha1}, Optimal alpha2: {self.alpha2}')
        return self.alpha1, self.alpha2

    def optimize_alpha_iter(self, X, y):
        def objective(params):
            alpha1, alpha2 = params
            print(f'Trying with alpha1: {alpha1}, alpha2: {alpha2}')
            scores = self.aggregate_scores_iter_batch(X, alpha1, alpha2)
            f1 = f1_score(y, scores > 0.5, average='macro')
            return -f1
        bounds = [(0, 1), (0, 1)]
        result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, popsize=30, mutation=(0.5, 1), recombination=0.8)
        self.alpha1, self.alpha2 = result.x
        print(f'Optimal alpha1: {self.alpha1}, Optimal alpha2: {self.alpha2}')
        return self.alpha1, self.alpha2

    def optimize_threshold(self, scores, y_true):
        def objective(threshold):
            y_pred = scores > threshold
            f1 = f1_score(y_true, y_pred, average='macro')
            return -f1
        print(f'min(socres): {np.min(scores)}')
        print(f'max(scores): {np.max(scores)}')
        result = minimize_scalar(objective, bounds=(np.min(scores), np.max(scores)), method='bounded')
        optimal_threshold = result.x
        print(f'Optimal threshold: {optimal_threshold}')

        best_threshold = optimal_threshold
        best_f1 = -result.fun
        for variation in np.linspace(-0.1, 0.1, 21):
            new_threshold = optimal_threshold + variation
            y_pred = scores > new_threshold
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = new_threshold
        print(f'Best threshold after variations: {best_threshold}')
        return best_threshold

    def evaluate_predictions(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Accuracy: {accuracy:.4f}')

    def plot_precision_recall_curve(self, y_true, scores):
        precision, recall, _ = precision_recall_curve(y_true == 1, scores)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()

    def fit(self, X, y):
        """
        Entrena el modelo generando el bosque de árboles usando hiTree_batch para evitar hojas vacías.
        """
        self.forest = []
        # Genera subconjuntos para cada árbol del bosque
        subsets = self.generate_subsets(X)
        for i, indices in enumerate(subsets):
            X_subset = X[indices]
            # Usa hiTree_batch para construir cada árbol con una semilla aleatoria diferente
            tree = self.hiTree_iterative(X_subset, SLab= np.zeros(X_subset.shape[0]), randSeed=i)
            self.forest.append(tree)

    def predict(self, X):
        scores = self.aggregate_scores_iter_batch(X, self.alpha1, self.alpha2)
        return scores

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

def generate_example_dataset():
    X_normal, _ = make_blobs(n_samples=3000, centers=[[0, 0]], cluster_std=1.0, random_state=42)
    X_normal_Labs = np.zeros(X_normal.shape[0])

    X_anomalies, _ = make_blobs(n_samples=50, centers=[[10, 10]], cluster_std=1.0, random_state=42)
    X_anomalies_Labs = np.ones(X_anomalies.shape[0])

    X = np.vstack([X_normal, X_anomalies])
    labs = np.hstack([X_normal_Labs, X_anomalies_Labs])

    return X, labs

def evaluate_predictions(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def plot_predictions(X, y_pred, title):
    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', label='Predicted Normal')
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', label='Predicted Anomalies')
    plt.title(title)
    plt.legend()
    plt.show()

def compare_algorithms(X, y):
    results = []

    hif = HybridIsolationForest(t=10, psi=64)

    start_time = time.time()
    mem_usage = memory_usage((hif.fit, (X, y)))
    train_time = time.time() - start_time
    train_memory = max(mem_usage) - min(mem_usage)

    start_time = time.time()
    mem_usage = memory_usage((hif.predict, (X,)))
    scores = hif.predict(X)
    pred_time = time.time() - start_time
    pred_memory = max(mem_usage) - min(mem_usage)

    threshold = hif.optimize_threshold(scores, y)
    y_pred = scores > threshold
    precision, recall, f1, accuracy = evaluate_predictions(y, y_pred)

    results.append({
        'Algorithm': 'Hybrid Isolation Forest',
        'Train Time': train_time,
        'Train Memory': train_memory,
        'Pred Time': pred_time,
        'Pred Memory': pred_memory,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
    })

    plot_confusion_matrix(y, y_pred, 'Confusion Matrix: Hybrid Isolation Forest')
    plot_predictions(X, y_pred, 'Predictions: Hybrid Isolation Forest')

    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

    start_time = time.time()
    mem_usage = memory_usage((iso_forest.fit, (X,)))
    train_time = time.time() - start_time
    train_memory = max(mem_usage) - min(mem_usage)

    start_time = time.time()
    mem_usage = memory_usage((iso_forest.predict, (X,)))
    y_pred = iso_forest.predict(X)
    y_pred = np.where(y_pred == -1, 1, 0)
    pred_time = time.time() - start_time
    pred_memory = max(mem_usage) - min(mem_usage)

    precision, recall, f1, accuracy = evaluate_predictions(y, y_pred)

    results.append({
        'Algorithm': 'Isolation Forest',
        'Train Time': train_time,
        'Train Memory': train_memory,
        'Pred Time': pred_time,
        'Pred Memory': pred_memory,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
    })

    plot_confusion_matrix(y, y_pred, 'Confusion Matrix: Isolation Forest')
    plot_predictions(X, y_pred, 'Predictions: Isolation Forest')

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)

    start_time = time.time()
    mem_usage = memory_usage((lof.fit_predict, (X,)))
    y_pred = lof.fit_predict(X)
    y_pred = np.where(y_pred == -1, 1, 0)
    pred_time = time.time() - start_time
    pred_memory = max(mem_usage) - min(mem_usage)

    precision, recall, f1, accuracy = evaluate_predictions(y, y_pred)

    results.append({
        'Algorithm': 'Local Outlier Factor',
        'Train Time': 0,
        'Train Memory': 0,
        'Pred Time': pred_time,
        'Pred Memory': pred_memory,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
    })

    plot_confusion_matrix(y, y_pred, 'Confusion Matrix: Local Outlier Factor')
    plot_predictions(X, y_pred, 'Predictions: Local Outlier Factor')

    results_df = pd.DataFrame(results)
    print(results_df)
