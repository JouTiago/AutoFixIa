import logging
from model_training import MLTrainer

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingController:
    def __init__(self, data_path="enriched_chat_data.csv"):
        # Inicializa o treinador com o caminho do dataset
        self.trainer = MLTrainer(data_path=data_path)
        # Executa o pipeline de treinamento automaticamente ao instanciar a classe
        self.run_training()

    def run_training(self):
        try:
            logger.info("Iniciando o pipeline de treinamento...")
            # Executa o pipeline de treinamento completo
            self.trainer.run_training_pipeline()
            logger.info("Pipeline de treinamento finalizado com sucesso.")
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {e}")

# Para executar a classe diretamente no terminal
# Crie uma instância da controller
controller = TrainingController()
