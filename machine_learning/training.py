import cx_Oracle
import pandas as pd
import json
from chat_bot.utils import get_db_connection
import logging
import datetime

logger = logging.getLogger(__name__)


class ChatDataProcessor:
    @staticmethod
    def consolidate_and_save_chat_logs():
        connection = get_db_connection()
        if connection is None:
            logger.error("Erro ao estabelecer conexão com o banco de dados para consolidar logs.")
            return
        try:
            with connection.cursor() as cursor:
                # Verifica quais chats já foram consolidados
                cursor.execute("SELECT id_chat FROM T_CHATBOT WHERE resposta_final IS NOT NULL")
                processed_chats = {row[0] for row in cursor.fetchall()}  # Conjunto de ids já processados

                # Busca todas as mensagens
                cursor.execute("SELECT id_chat, mensagem FROM T_MENSAGENS ORDER BY id_chat, id_mensagem")
                messages = cursor.fetchall()

                consolidated_logs = {}
                for id_chat, message in messages:
                    # Ignora chats já processados
                    if id_chat in processed_chats:
                        continue

                    # Converte o LOB para string, se necessário
                    if isinstance(message, cx_Oracle.LOB):
                        message = message.read()

                    if id_chat not in consolidated_logs:
                        consolidated_logs[id_chat] = []
                    consolidated_logs[id_chat].append(message)

                # Salva o JSON consolidado para cada chat_id não processado
                for chat_id, log in consolidated_logs.items():
                    full_conversation = json.dumps(log)
                    update_params = {
                        'resposta_final': full_conversation,
                        'resposta_data': datetime.datetime.now(),
                        'id_chat': chat_id
                    }
                    cursor.execute("""
                            UPDATE T_CHATBOT
                            SET resposta_final = :resposta_final, resposta_data = :resposta_data
                            WHERE id_chat = :id_chat
                        """, update_params)
                connection.commit()
                logger.info("Logs consolidados e salvos com sucesso em T_CHATBOT.")
        except cx_Oracle.Error as e:
            logger.error(f"Erro ao consolidar e salvar logs de chat: {e}")
        finally:
            connection.close()

    @staticmethod
    def export_chat_data_to_csv(output_path="chat_data.csv"):
        # Abre uma nova conexão para exportar os dados
        connection = get_db_connection()
        if connection is None:
            logger.error("Erro ao estabelecer conexão com o banco de dados para exportar dados.")
            return
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM T_CHATBOT")
                columns = [col[0] for col in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(output_path, index=False)
                logger.info(f"Dados de T_CHATBOT exportados para {output_path}.")
        except cx_Oracle.Error as e:
            logger.error(f"Erro ao exportar dados de T_CHATBOT para CSV: {e}")
        finally:
            connection.close()