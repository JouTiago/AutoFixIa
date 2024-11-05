from chat_bot import create_app  # Supondo que create_app esteja configurado
from chat_bot.utils import get_db_connection
from machine_learning.llm_increment import ConversationSimulator
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Caminhos dos arquivos CSV
csv_path = "consolidated_chat_data.csv"
enriched_csv_path = "enriched_chat_data.csv"

# Inicializar a aplicação Flask para criar o contexto
app = create_app()

with app.app_context():
    # Conecta ao banco de dados
    connection = get_db_connection()

    logging.info("Rodando simulações")
    ConversationSimulator.simulate_conversation()

    # Enriquecer o CSV com dados simulados
    logging.info("Enriquecendo dados com simulações...")
    ConversationSimulator.enrich_and_save_chat_data(csv_path=csv_path, output_path=enriched_csv_path)

    # Fecha a conexão com o banco de dados
    connection.close()

logging.info("Pipeline de dados concluído com sucesso.")
