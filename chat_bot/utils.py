from functools import wraps
from flask import request, jsonify, g
import cx_Oracle
import logging
from .config import Config

logger = logging.getLogger(__name__)


def init_oracle_pool():
    try:
        pool = cx_Oracle.SessionPool(
            user=Config.ORACLE_USER,
            password=Config.ORACLE_PASSWORD,
            dsn=cx_Oracle.makedsn(Config.ORACLE_HOST, Config.ORACLE_PORT, service_name=Config.ORACLE_SERVICE_NAME),
            min=2,
            max=10,
            increment=1,
            encoding="UTF-8"
        )
        return pool
    except cx_Oracle.Error as e:
        logger.error(f"Erro ao criar o pool de sessões Oracle: {e}")
        return None


oracle_pool = init_oracle_pool()


def get_db_connection():
    global oracle_pool
    if 'db_connection' not in g:
        if oracle_pool:
            try:
                g.db_connection = oracle_pool.acquire()
                logger.info("Conexão Oracle adquirida do pool.")
            except cx_Oracle.Error as e:
                logger.error(f"Erro ao adquirir conexão do pool: {e}")
                g.db_connection = None
        else:
            logger.error("Pool de sessões Oracle não está disponível.")
            g.db_connection = None
    return g.db_connection


def verificar_service_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        service_token = request.headers.get('Service-Token')
        if not service_token or service_token != Config.SERVICE_TOKEN:
            logger.warning("Service-Token inválido ou ausente.")
            return jsonify({'error': 'Service-Token inválido ou ausente.'}), 401
        return f(*args, **kwargs)

    return decorated


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        data = request.get_json()
        token = data.get('Service-Token')
        if not token:
            logger.warning("Token de autenticação ausente.")
            return jsonify({'error': 'Token de autenticação é necessário.'}), 401
        if token != Config.SECRET_KEY:
            logger.warning("Token de autenticação inválido.")
            return jsonify({'error': 'Token inválido.'}), 401
        user_id = "usuario_padrao"
        return f(user_id, *args, **kwargs)
    return decorated


def get_pdf_blob_content(id_manual):
    """
    Busca o conteúdo do PDF como BLOB do banco de dados com base no ID do manual.
    """
    connection = get_db_connection()
    if not connection:
        print("Erro ao conectar ao banco de dados.")
        return None

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT pdf_manual FROM T_MANUAL WHERE id_manual = :id_manual
            """, id_manual=id_manual)
            result = cursor.fetchone()
            if result:
                pdf_blob = result[0].read()
                return pdf_blob
            else:
                print("Manual não encontrado para o ID fornecido.")
                return None
    except cx_Oracle.Error as e:
        print(f"Erro ao buscar o PDF do banco: {e}")
        return None
    finally:
        connection.close()
