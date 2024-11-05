import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import os

logger = logging.getLogger(__name__)


class MLTrainer:
    def __init__(self, data_path="enriched_chat_data.csv"):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.data = self.prepare_dataset(self.data)

    def prepare_dataset(self, data):
        # Limpeza e criação de novas colunas
        data.dropna(subset=['complexity', 'resolved_status'], inplace=True)
        data['conversation'].fillna('', inplace=True)
        data['length'] = data['conversation'].apply(len)
        data['num_messages'] = data['conversation'].apply(lambda x: x.count("user") + x.count("bot"))
        data['duration'] = data['conversation'].apply(lambda x: np.random.randint(1, 10))

        # Balanceamento das classes
        majority_class = data[data['resolved_status'] == 'Resolvido']
        minority_class = data[data['resolved_status'] == 'Assistência Manual']
        minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class),
                                            random_state=42)
        data_balanced = pd.concat([majority_class, minority_class_upsampled])

        logger.info("Dataset balanceado para o treinamento.")
        return data_balanced

    def train_regression_model(self):
        X = self.data[['length', 'num_messages', 'duration']]
        y = self.data['complexity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg_model = Ridge()
        grid_search = GridSearchCV(reg_model, {'alpha': [0.1, 1.0, 10.0]}, cv=5)
        grid_search.fit(X_train, y_train)
        best_reg_model = grid_search.best_estimator_

        with open("regression_model.pkl", "wb") as f:
            pickle.dump(best_reg_model, f)

        y_pred = best_reg_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"MSE da Regressão: {mse}")

        # Adiciona as previsões de complexidade ao DataFrame
        self.data["complexity_pred"] = best_reg_model.predict(X)

    def train_classification_model(self):
        X = self.data[['length', 'num_messages', 'duration', 'complexity_pred']]
        y = self.data['resolved_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cls_model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(cls_model, {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.5]}, cv=5)
        grid_search.fit(X_train, y_train)
        best_cls_model = grid_search.best_estimator_

        with open("classification_model.pkl", "wb") as f:
            pickle.dump(best_cls_model, f)

        y_pred = best_cls_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        logger.info(f"Acurácia: {accuracy}, Precisão: {precision}, Recall: {recall}, F1-Score: {f1}")

    def run_training_pipeline(self):
        logger.info("Iniciando pipeline de treinamento...")
        self.train_regression_model()
        self.train_classification_model()
        self.save_enriched_data()
        logger.info("Pipeline de treinamento completo.")

    def save_enriched_data(self, output_path="enriched_chat_data_with_predictions.csv"):
        self.data.to_csv(output_path, index=False)
        logger.info(f"Dataset enriquecido salvo em {output_path}.")
