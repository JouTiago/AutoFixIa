import cx_Oracle
from flask import Blueprint, request, jsonify
from chat_bot.utils import verificar_service_token, token_required, get_db_connection
import uuid
import logging
from chat_bot.process import initialize_rag_with_manual, generate_query_with_control, chunk_embeddings

main = Blueprint('main', __name__)

logger = logging.getLogger(__name__)


@main.route('/chat/init', methods=['POST'])
@verificar_service_token
def init_chat():
    logger.info("Requisição recebida em /chat/init")
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Não foi possível conectar ao banco de dados.'}), 500

    data = request.get_json()
    veiculo = data.get('veiculo')
    cpf = request.headers.get('Cpf')
    logger.info(f"CPF recebido: {cpf}")

    if not veiculo or not cpf:
        return jsonify({'error': 'Dados do veículo e CPF são necessários.'}), 400

    try:
        with connection.cursor() as cursor:
            params_cliente = {'c_cpf': cpf}
            cursor.execute("""
                SELECT * FROM T_CLIENTE WHERE TRIM(c_cpf) = TRIM(:c_cpf)
            """, params_cliente)
            result_cliente = cursor.fetchone()
            if result_cliente:
                logger.info(f"Cliente encontrado: {result_cliente}")
            else:
                logger.info("Cliente não encontrado.")

            params = {
                'marca': veiculo['marca'],
                'modelo': veiculo['modelo'],
                'ano': str(veiculo['ano'])
            }
            cursor.execute("""
                SELECT id_manual FROM T_MANUAL
                WHERE marca_manual = :marca AND modelo_manual = :modelo AND ano_manual = :ano
            """, params)
            result = cursor.fetchone()

            if not result:
                return jsonify({'error': 'Manual não encontrado para o veículo especificado.'}), 404

            id_manual = result[0]
            id_chat = str(uuid.uuid4())

            params_chatbot = {
                'id_chat': id_chat,
                'id_manual': id_manual,
                'c_cpf': cpf
            }
            cursor.execute("""
                INSERT INTO T_CHATBOT (id_chat, resposta_final, resposta_data, id_manual, c_cpf)
                VALUES (:id_chat, NULL, NULL, :id_manual, :c_cpf)
            """, params_chatbot)
            connection.commit()

            initialize_rag_with_manual(id_manual)

            logger.info(f"Sessão de chat {id_chat} iniciada para o veículo {veiculo}.")
            return jsonify({'chat_id': id_chat}), 200

    except cx_Oracle.Error as e:
        logger.error(f"Erro ao iniciar chat: {e}")
        return jsonify({'error': f'Erro ao iniciar chat: {str(e)}'}), 500


@main.route('/chat/send', methods=['POST'])
@verificar_service_token
@token_required
def send_message(user_id):
    logger.info("Requisição recebida em /chat/send")
    token = request.headers.get('Service-Token')
    logger.info(f"Token recebido: {token}")
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Não foi possível conectar ao banco de dados.'}), 500

    try:
        data = request.get_json()
        chat_id = data.get('chatId')
        message = data.get('mensagem')
        logger.info(f" Chat Id: {chat_id}")
        logger.info(f" mESSAGE: {message}")

        if not chat_id or not message:
            return jsonify({'error': 'chat_id e mensagem são obrigatórios'}), 400

        with connection.cursor() as cursor:
            params = {'id_chat': chat_id}
            logger.info(f" Params: {params}")
            cursor.execute("""
                SELECT id_manual FROM T_CHATBOT WHERE id_chat = :id_chat
            """, params)
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'Sessão de chat não encontrada ou acesso negado.'}), 403

            id_manual = result[0]

            if not chunk_embeddings:
                initialize_rag_with_manual(id_manual)

            cursor.execute("SELECT seq_id_mensagem.NEXTVAL FROM dual")
            id_mensagem_usuario = cursor.fetchone()[0]
            logger.info(f"ID msg usuario: {id_mensagem_usuario}")

            params_usuario = {
                'id_mensagem': id_mensagem_usuario,
                'id_chat': chat_id,
                'remetente': 'usuario',
                'mensagem': message
            }
            cursor.execute("""
                INSERT INTO T_MENSAGENS (id_mensagem, id_chat, remetente, mensagem)
                VALUES (:id_mensagem, :id_chat, :remetente, :mensagem)
            """, params_usuario)
            connection.commit()
            logger.info("inserido no banco")

            response = generate_query_with_control(chat_id, message)
            logger.info(f"Response: {response}")

            cursor.execute("SELECT seq_id_mensagem.NEXTVAL FROM dual")
            id_mensagem_bot = cursor.fetchone()[0]

            params_bot = {
                'id_mensagem': id_mensagem_bot,
                'id_chat': chat_id,
                'remetente': 'bot',
                'mensagem': response
            }
            cursor.execute("""
                INSERT INTO T_MENSAGENS (id_mensagem, id_chat, remetente, mensagem)
                VALUES (:id_mensagem, :id_chat, :remetente, :mensagem)
            """, params_bot)
            connection.commit()

            logger.info(f"Mensagem enviada na sessão {chat_id} pelo usuário {user_id}.")
            return jsonify({'response': response, 'user_id': user_id, 'id_chat': chat_id}), 200
    except cx_Oracle.Error as e:
        logger.error(f"Erro ao enviar mensagem: {e}")
        return jsonify({'error': 'Erro ao enviar mensagem.'}), 500
    finally:
        connection.close()


@main.route('/chat/end', methods=['POST'])
@verificar_service_token
@token_required
def end_chat(user_id):
    logger.info("Requisição recebida em /chat/end")
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Não foi possível conectar ao banco de dados.'}), 500
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')

        if not chat_id:
            return jsonify({'error': 'chat_id é obrigatório'}), 400

        with connection.cursor() as cursor:
            params = {'id_chat': chat_id}
            cursor.execute("""
                SELECT c_cpf FROM T_CHATBOT WHERE id_chat = :id_chat
            """, params)
            result = cursor.fetchone()
            if not result or result[0] != user_id:
                return jsonify({'error': 'Sessão de chat não encontrada ou acesso negado.'}), 403

            logger.info(f"Sessão de chat {chat_id} encerrada pelo usuário {user_id}.")
            return jsonify({'message': 'Sessão de chat encerrada com sucesso.'}), 200
    except cx_Oracle.Error as e:
        logger.error(f"Erro ao encerrar chat: {e}")
        return jsonify({'error': 'Erro ao encerrar chat.'}), 500
    finally:
        connection.close()
