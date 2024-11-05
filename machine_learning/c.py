from model_predict import ModelPredict
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import logging

# Configuração do logger para garantir a saída dos logs no console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PredictionController:
    def __init__(self):
        logger.info("Iniciando o controlador de previsão...")
        self.predictor = ModelPredict()
        self.run_predictions()

    def run_predictions(self):
        logger.info("Executando previsões...")

        # Definindo valores de entrada para a previsão
        num_messages = 15
        length = 120  # Comprimento da conversa em caracteres
        duration = 5  # Duração da conversa em minutos

        # Previsão de complexidade
        complexity_pred = self.predictor.predict_complexity(num_messages, length, duration)
        if complexity_pred is not None:
            logger.info(f"Complexidade Prevista: {complexity_pred}")

            # Previsão de status usando a complexidade prevista
            status_pred = self.predictor.predict_status(num_messages, length, duration, complexity_pred)
            if status_pred is not None:
                logger.info(f"Status Previsto: {status_pred}")
            else:
                logger.error("Erro ao prever o status.")
        else:
            logger.error("Erro ao prever a complexidade.")

# Inicializar e executar a PredictionController diretamente
PredictionController()

# y_true_reg e y_pred_reg são para regressão; y_true_cls e y_pred_cls são para classificação
y_true_reg = [3.5, 2.0, 4.2]  # Valores reais de complexidade
y_pred_reg = [3.6, 1.9, 4.1]  # Valores previstos de complexidade pelo modelo

# Para regressão
mse = mean_squared_error(y_true_reg, y_pred_reg)
print("Erro Quadrático Médio (MSE) da Regressão:", mse)

# Para classificação, usando y_true_cls e y_pred_cls
y_true_cls = ["Resolvido", "Assistência Manual", "Resolvido"]
y_pred_cls = ["Resolvido", "Resolvido", "Resolvido"]

accuracy = accuracy_score(y_true_cls, y_pred_cls)
precision = precision_score(y_true_cls, y_pred_cls, average='weighted')
recall = recall_score(y_true_cls, y_pred_cls, average='weighted')
f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')

print("Acurácia da Classificação:", accuracy)
print("Precisão:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
