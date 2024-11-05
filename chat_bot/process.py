import logging
import threading
import uuid
from typing import List, Tuple
import json
import os

import fitz
from collections import defaultdict
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import SystemMessage, HumanMessage
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from .config import Config
from .utils import get_pdf_blob_content

logger = logging.getLogger(__name__)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=AzureOpenAIEmbeddings(),
    persist_directory="./chroma_langchain_db",
)

store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vector_store,
    docstore=store,
    id_key=id_key,
)

ai_model = AzureChatOpenAI(
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    deployment_name=Config.AZURE_CHAT_DEPLOYMENT,
    api_version="2024-02-15-preview",
    max_tokens=4096,
    max_retries=5
)

chat_sessions = {}
chat_question_count = defaultdict(int)
chat_interactions = defaultdict(list)
lock = threading.Lock()
chunks = []
chunk_embeddings = []
diagnosis_pending = {}
PROCESSED_MANUALS_FILE = "processed_manuals.json"


def generate_query_with_control(chat_id: str, query: str) -> str:
    global chat_question_count, diagnosis_pending

    with lock:
        chat_question_count[chat_id] += 1
        chat_interactions[chat_id].append(query)

    if diagnosis_pending.get(chat_id):
        diagnosis_pending.pop(chat_id, None)
        response = generate_final_diagnosis(chat_id)
        chat_question_count.pop(chat_id, None)
        chat_interactions.pop(chat_id, None)
        filter_response(response)
        response = response.replace("\n", "\\n").replace("\t", "\\t")
        return response

    if chat_question_count[chat_id] >= 5:
        diagnosis_pending[chat_id] = True
        return process_query(query)

    return process_query(query)


def generate_final_diagnosis(chat_id: str) -> str:
    """
    Gera um diagnóstico final com base nas interações realizadas.
    """
    user_questions = chat_interactions[chat_id][::2]
    bot_responses = chat_interactions[chat_id][1::2]

    diagnosis_context = "\n".join(
        f"Pergunta {i + 1}: {user_questions[i]}\nResposta: {bot_responses[i]}"
        for i in range(len(bot_responses))
    )

    final_response = ai_model([
        SystemMessage(content="""Você é um assistente especializado em diagnósticos automotivos.
        Use as informações a seguir para gerar um diagnóstico final baseado nas perguntas e respostas anteriores, 
        considerando todas as interações com o usuário."""),
        HumanMessage(content=f"Contexto:\n{diagnosis_context}\n\nDiagnóstico final:")
    ])

    return final_response.content.strip()


def process_query(query: str) -> str:
    global chunks, chunk_embeddings

    embeddings_model = AzureOpenAIEmbeddings()
    query_embedding = embeddings_model.embed_query(query)

    similarities = [compute_cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]

    if not similarities:
        logger.warning("Nenhuma similaridade calculada. Gerando pergunta esclarecedora.")
        return generate_clarifying_question(query)

    max_similarity = max(similarities)
    mean_similarity = sum(similarities) / len(similarities)
    similarity_diff = max_similarity - mean_similarity

    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:Config.TOP_K]

    if max_similarity >= Config.SIMILARITY_THRESHOLD and similarity_diff >= Config.SIMILARITY_DIFF_THRESHOLD:
        similar_chunks = [chunks[i] for i in top_k_indices]
        relevant_text = "\n".join(similar_chunks)

        response = ai_model([
            SystemMessage(content="""Você é um assistente especializado que responde com base no manual do carro 
            fornecido. Responda de forma detalhada e específica, evitando recomendações gerais como levar a um mecânico.
            Após cinco perguntas, forneça um diagnóstico final."""),
            HumanMessage(content=f"Contexto:\n{relevant_text}\n\nPergunta: {query}\n\nResposta:")
        ])

        filtered_response = filter_response(response.content.strip())

        if not filtered_response or len(filtered_response) < 50:
            clarifying_question = generate_clarifying_question(query)
            return clarifying_question
        else:
            return filtered_response
    else:
        clarifying_question = generate_clarifying_question(query)
        return clarifying_question


def generate_clarifying_question(query: str) -> str:
    """
    Gera uma pergunta esclarecedora baseada na consulta original do usuário.
    """
    prompt_text = f"""O usuário perguntou: "{query}". A resposta anterior não está clara ou foi suficiente para 
    identificar o problema com base no manual do carro. Com base nisso, faça uma pergunta direcionada e específica
    para obter mais informações que possam ajudar a resolver o problema do usuário."""

    prompt = ChatPromptTemplate.from_template("{prompt}")

    response = ai_model([
        SystemMessage(
            content="Você é um assistente útil que faz perguntas esclarecedoras para obter mais informações."),
        HumanMessage(content=prompt_text)
    ])

    return response.content.strip()


def filter_response(response: str) -> str:
    """
    Filtra a resposta para remover recomendações gerais como "levar a um mecânico".
    """
    phrases_to_remove = [
        "recomenda-se levar o carro a um mecânico",
        "é altamente recomendável levar seu carro a um profissional",
        "levar a um mecânico especializado",
        "procure um mecânico",
        "levar a um profissional qualificado",
        "considere levar o veículo a uma oficina especializada",
        "encaminhe o carro para um profissional"
    ]

    for phrase in phrases_to_remove:
        if phrase in response.lower():
            response = response.replace(phrase, "")

    return response.strip()


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula a similaridade do cosseno entre dois vetores.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def extract_elements_pymupdf(pdf_blob: bytes) -> Tuple[List[dict], List[dict]]:
    """
    Extrai texto e tabelas de um PDF usando PyMuPDF a partir de um BLOB de bytes.
    """
    extracted_table_elements = []
    extracted_text_elements = []
    try:
        doc = fitz.open("pdf", pdf_blob)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                extracted_text_elements.append({"type": "text", "text": text})
        logger.info(f"Total de textos extraídos com PyMuPDF: {len(extracted_text_elements)}")
    except Exception as e:
        logger.error(f"Erro ao extrair elementos com PyMuPDF: {e}")
    return extracted_table_elements, extracted_text_elements


def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Divide o texto em chunks de tamanho especificado.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def element_summarizer(table_elements: List[dict], text_elements: List[dict], model: AzureChatOpenAI):
    prompt_text = """Você é um assistente encarregado de resumir tabelas e textos. 
    Forneça um resumo conciso da tabela ou do texto. Tabela ou trecho de texto: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    texts = [e["text"] for e in text_elements if e["text"].strip() != ""]
    if texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    else:
        text_summaries = []

    tables = [e["text"] for e in table_elements]
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    else:
        table_summaries = []

    return table_summaries, text_summaries


def indextexts(texts: List[str], text_summaries: List[str], retriever_instance: MultiVectorRetriever,
               identifier_key: str = "doc_id"):
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={identifier_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever_instance.vectorstore.add_documents(summary_texts)
    retriever_instance.docstore.mset(list(zip(doc_ids, texts)))


def load_processed_manuals():
    if os.path.exists(PROCESSED_MANUALS_FILE):
        with open(PROCESSED_MANUALS_FILE, "r") as file:
            data = json.load(file)
            return data.get("processed_manuals", [])
    else:
        with open(PROCESSED_MANUALS_FILE, "w") as file:
            json.dump({"processed_manuals": []}, file)
        return []


def save_processed_manual(id_manual):
    processed_manuals = load_processed_manuals()
    if id_manual not in processed_manuals:
        processed_manuals.append(id_manual)
        with open(PROCESSED_MANUALS_FILE, "w") as file:
            json.dump({"processed_manuals": processed_manuals}, file)


def initialize_rag_with_manual(id_manual: int):
    processed_manuals = load_processed_manuals()
    if id_manual in processed_manuals:
        logger.info("Manual já processado anteriormente. Usando dados indexados.")
        return

    pdf_blob = get_pdf_blob_content(id_manual)
    if pdf_blob:
        logger.info("Iniciando o RAG com o manual do carro.")

        table_elements, text_elements = extract_elements_pymupdf(pdf_blob)
        table_summaries, text_summaries = element_summarizer(table_elements, text_elements, ai_model)
        texts = [e["text"] for e in text_elements]

        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={"doc_id": str(id_manual)})
            for i, s in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)

        global chunks, chunk_embeddings
        chunks = split_text_into_chunks(" ".join(texts), Config.CHUNK_SIZE)
        embeddings_model = AzureOpenAIEmbeddings()
        chunk_embeddings = embeddings_model.embed_documents(chunks)

        save_processed_manual(id_manual)
        logger.info("Inicialização do RAG concluída para o manual especificado.")
    else:
        logger.error("Não foi possível encontrar o manual especificado para inicializar o RAG.")
