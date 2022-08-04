from typing import Tuple
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def classificador_flores_iris(train_input_features: np.ndarray, train_outputs: np.ndarray, forecast_features: np.ndarray) -> np.ndarray:
    classificador = KNeighborsClassifier(n_neighbors=3)
    
    classificador.fit(train_input_features, train_outputs)
    result = classificador.predict(forecast_features)

    return result
