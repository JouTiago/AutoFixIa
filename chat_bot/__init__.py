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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.getLogger(__name__)

    app.register_blueprint(main_blueprint)

    return app
