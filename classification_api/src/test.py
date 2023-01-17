import numpy as np
from sklearn.metrics import roc_auc_score
import json
import requests

from definitions import DATA_DIR

X_test = np.load(DATA_DIR / 'X_test.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')
y_pred = []

for x in X_test:
    data = {"request": x}
    response = requests.get('http://127.0.0.1:80/predict', params=data)
    y_pred.append(json.loads(response.text)['disease'])

print(f"ROC AUC score: {roc_auc_score(y_test, y_pred)}")
