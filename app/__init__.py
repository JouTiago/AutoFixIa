import cx_Oracle
from flask import Flask
from flask_cors import CORS
from .config import Config
import logging
from .routes import main as main_blueprint
from .utils import oracle_pool


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    # Configuração de Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    # Registrar Blueprints ou Routes
    app.register_blueprint(main_blueprint)

    # Encerrar o pool de sessões quando a aplicação for encerrada
    # @app.teardown_appcontext
    # def shutdown_session(exc=None):
    #     global oracle_pool
    #     if oracle_pool is not None:
    #         try:
    #             oracle_pool.close()
    #             logger.info("Pool de sessões Oracle fechado.")
    #         except cx_Oracle.Error as e:
    #             logger.error(f"Erro ao fechar o pool de sessões Oracle: {e}")
    return app
