from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Configurações
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
    app.config['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
    # Adicione outras configurações conforme necessário

    # Registrar Blueprints ou Routes
    from .routes import main
    app.register_blueprint(main)

    return app
