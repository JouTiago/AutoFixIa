import cx_Oracle
from flask import Blueprint, request, jsonify
from .utils import verificar_service_token, token_required, get_db_connection
import uuid
import logging
from .process import process_query, initialize_rag_with_manual

main = Blueprint('main', __name__)

logger = logging.getLogger(__name__)


@main.route('/chat/init', methods=['POST'])
@verificar_service_token
@token_required
def init_chat(user_id):
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Não foi possível conectar ao banco de dados.'}), 500

    data = request.get_json()
    veiculo = data.get('veiculo')  # Dados do veículo, contendo marca, modelo e ano
    if not veiculo:
        return jsonify({'error': 'Dados do veículo são necessários.'}), 400

    try:
        # Buscar o manual correspondente no banco de dados
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id_manual FROM T_MANUAL
                WHERE marca_manual = :marca AND modelo_manual = :modelo AND ano_manual = :ano
            """, marca=veiculo['marca'], modelo=veiculo['modelo'], ano=veiculo['ano'])
            result = cursor.fetchone()

            if not result:
                return jsonify({'error': 'Manual não encontrado para o veículo especificado.'}), 404

            id_manual = result[0]
            id_chat = str(uuid.uuid4())

            # Inserir novo registro em T_CHATBOT
            cursor.execute("""
                INSERT INTO T_CHATBOT (id_chat, resposta_final, resposta_data, id_manual, c_cpf)
                VALUES (:uid, NULL, NULL, :id_manual, :cpf)
            """, uid=id_chat, id_manual=id_manual, cpf=user_id)
            connection.commit()

            # Inicializar o RAG com o manual correspondente
            initialize_rag_with_manual(id_manual)

            # Mensagem inicial
            initial_message = ("Olá, sou o chatbot da Autofix e vou te ajudar a identificar quaisquer problemas que "
                               "você possa estar enfrentando com o seu veículo. Qual problema você está enfrentando?")
            cursor.execute("""
                INSERT INTO T_MENSAGENS (id_chat, remetente, mensagem)
                VALUES (:uid, 'assistant', :msg)
            """, uid=id_chat, msg=initial_message)
            connection.commit()

            logger.info(f"Sessão de chat {id_chat} iniciada para o veículo {veiculo}.")
            return jsonify({'chat_id': id_chat, 'initial_message': initial_message}), 200

    except cx_Oracle.Error as e:
        logger.error(f"Erro ao iniciar chat: {e}")
        return jsonify({'error': 'Erro ao iniciar chat.'}), 500
    finally:
        connection.close()


@main.route('/chat/send', methods=['POST'])
@verificar_service_token
@token_required
def send_message(user_id):
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Não foi possível conectar ao banco de dados.'}), 500
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')
        message = data.get('mensagem')

        if not chat_id or not message:
            return jsonify({'error': 'chat_id e mensagem são obrigatórios'}), 400

        with connection.cursor() as cursor:
            # Verificar se a sessão de chat pertence ao usuário
            cursor.execute("""
                SELECT c_cpf FROM T_CHATBOT WHERE id_chat = :uid
            """, uid=chat_id)
            result = cursor.fetchone()
            if not result or result[0] != user_id:
                return jsonify({'error': 'Sessão de chat não encontrada ou acesso negado.'}), 403

            # Inserir mensagem do usuário na tabela T_MESSAGES
            cursor.execute("""
                INSERT INTO T_MESSAGES (id_chat, sender, message)
                VALUES (:uid, 'user', :msg)
            """, uid=chat_id, msg=message)
            connection.commit()

            # Processar a mensagem com o modelo RAG
            response = process_query(message)

            # Inserir resposta do assistente na tabela T_MESSAGES
            cursor.execute("""
                INSERT INTO T_MESSAGES (id_chat, sender, message)
                VALUES (:uid, 'assistant', :msg)
            """, uid=chat_id, msg=response)
            connection.commit()

            # Atualizar a tabela T_CHATBOT com a resposta final
            cursor.execute("""
                UPDATE T_CHATBOT
                SET resposta_final = :resp, resposta_data = CURRENT_TIMESTAMP
                WHERE id_chat = :uid
            """, resp=response, uid=chat_id)
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
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Não foi possível conectar ao banco de dados.'}), 500
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')

        if not chat_id:
            return jsonify({'error': 'chat_id é obrigatório'}), 400

        with connection.cursor() as cursor:
            # Verificar se a sessão de chat pertence ao usuário
            cursor.execute("""
                SELECT c_cpf FROM T_CHATBOT WHERE id_chat = :uid
            """, uid=chat_id)
            result = cursor.fetchone()
            if not result or result[0] != user_id:
                return jsonify({'error': 'Sessão de chat não encontrada ou acesso negado.'}), 403

            # Encerrar a sessão de chat (opcional: pode-se marcar como encerrada)
            cursor.execute("""
                DELETE FROM T_CHATBOT WHERE id_chat = :uid
            """, uid=chat_id)
            connection.commit()

            cursor.execute("""
                DELETE FROM T_MESSAGES WHERE id_chat = :uid
            """, uid=chat_id)
            connection.commit()

            logger.info(f"Sessão de chat {chat_id} encerrada pelo usuário {user_id}.")
            return jsonify({'message': 'Sessão de chat encerrada com sucesso.'}), 200
    except cx_Oracle.Error as e:
        logger.error(f"Erro ao encerrar chat: {e}")
        return jsonify({'error': 'Erro ao encerrar chat.'}), 500
    finally:
        connection.close()
