import os
from dotenv import load_dotenv
import cx_Oracle

# Carregar variáveis de ambiente do .env
load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret")
    SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "default-token")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT")
    AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")
    ORACLE_USER = os.getenv("ORACLE_USER")
    ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
    ORACLE_HOST = os.getenv("ORACLE_HOST")
    ORACLE_PORT = os.getenv("ORACLE_PORT")
    ORACLE_SERVICE_NAME = os.getenv("ORACLE_SERVICE_NAME")
    PDF_PATH = os.getenv("PDF_PATH")
    IMAGE_PATH = os.getenv("IMAGE_PATH")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    TOP_K = int(os.getenv("TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    SIMILARITY_DIFF_THRESHOLD = float(os.getenv("SIMILARITY_DIFF_THRESHOLD", 0.1))

def get_oracle_connection():
    dsn_tns = cx_Oracle.makedsn(Config.ORACLE_HOST, Config.ORACLE_PORT, service_name=Config.ORACLE_SERVICE_NAME)
    return cx_Oracle.connect(user=Config.ORACLE_USER, password=Config.ORACLE_PASSWORD, dsn=dsn_tns)
