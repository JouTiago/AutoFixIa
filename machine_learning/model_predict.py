import pickle
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import os

logger = logging.getLogger(__name__)

class ModelPredict:
    def __init__(self):
        self.reg_model_path = "regression_model.pkl"
        self.cls_model_path = "classification_model.pkl"
        self.reg_model = self.load_model(self.reg_model_path)
        self.cls_model = self.load_model(self.cls_model_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            return None
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def predict_complexity(self, num_messages, length, duration):
        if self.reg_model is None:
            logger.error("Modelo de regressão não carregado.")
            return None

        X = pd.DataFrame([[length, num_messages, duration]], columns=['length', 'num_messages', 'duration'])
        try:
            prediction = self.reg_model.predict(X)
            return prediction[0]
        except Exception as e:
            logger.error(f"Erro ao prever complexidade: {e}")
            return None

    def predict_status(self, num_messages, length, duration, complexity_pred):
        if self.cls_model is None:
            logger.error("Modelo de classificação não carregado.")
            return None

        X = pd.DataFrame([[length, num_messages, duration, complexity_pred]], columns=['length', 'num_messages', 'duration', 'complexity_pred'])
        try:
            prediction = self.cls_model.predict(X)
            return prediction[0]
        except Exception as e:
            logger.error(f"Erro ao prever status: {e}")
            return None

    def evaluate_models(self, test_data):
        if self.reg_model is None or self.cls_model is None:
            logger.error("Modelos de regressão ou classificação não carregados.")
            return None

        # Separando features e labels para avaliação
        X_test_reg = test_data[['length', 'num_messages', 'duration']]
        y_test_reg = test_data['complexity']
        X_test_cls = test_data[['length', 'num_messages', 'duration', 'complexity_pred']]
        y_test_cls = test_data['resolved_status']

        # Avaliação do modelo de regressão
        y_pred_reg = self.reg_model.predict(X_test_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        logger.info(f"Erro Quadrático Médio (MSE) da Regressão: {mse}")

        # Avaliação do modelo de classificação
        y_pred_cls = self.cls_model.predict(X_test_cls)
        accuracy = accuracy_score(y_test_cls, y_pred_cls)
        precision = precision_score(y_test_cls, y_pred_cls, average='weighted', zero_division=0)
        recall = recall_score(y_test_cls, y_pred_cls, average='weighted')
        f1 = f1_score(y_test_cls, y_pred_cls, average='weighted')

        logger.info(f"Acurácia da Classificação: {accuracy}")
        logger.info(f"Precisão: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1-Score: {f1}")

        return {
            "mse": mse,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
