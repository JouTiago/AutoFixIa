import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from chat_bot.process import ai_model
import json
import uuid
import random
from faker import Faker
import logging

# Configurações de logging
logging.basicConfig(level=logging.INFO)

fake = Faker('pt_BR')  # Define o Faker para português do Brasil


class ConversationSimulator:
    @staticmethod
    def simulate_conversation(num_samples=1):
        chat_data = []
        for i in range(num_samples):
            print(f"Loop de samples: {i}")
            problem_type = np.random.choice(["freio", "motor", "suspensão", "direção", "transmissão"])
            urgency_level = np.random.choice(["leve", "moderado", "grave"])
            user_knowledge = np.random.choice(["baixo", "médio", "alto"])

            # Mensagem inicial
            conversation = [{
                "role": "user",
                "content": ai_model([HumanMessage(
                    content=f"Estou tendo problemas com o {problem_type} do meu carro, urgência: "
                            f"{urgency_level}. Conhecimento: {user_knowledge}."
                )]).content
            }]
            print("Mensagem inicial gerada")

            # Loop para gerar uma sequência limitada de respostas e seguimentos
            for _ in range(3):
                print(f"Loop: {i}")  # Limita o número de iterações para evitar loops infinitos
                bot_response = ai_model([
                    SystemMessage(content="Responder ao usuário"),
                    HumanMessage(content=conversation[-1]["content"])
                ]).content
                user_followup = ai_model([
                    SystemMessage(content="Usuário descreve mais detalhes"),
                    HumanMessage(content=bot_response)
                ]).content

                conversation.append({"role": "bot", "content": bot_response})
                conversation.append({"role": "user", "content": user_followup})

            # Definir a complexidade e status de resolução
            complexity = np.random.randint(1, 5)
            resolved_status = "Resolvido" if complexity <= 3 else "Assistência Manual"
            print(resolved_status)

            # Adicionar a conversa simulada aos dados de chat
            chat_data.append({
                "conversation": json.dumps(conversation),
                "problem_type": problem_type,
                "urgency_level": urgency_level,
                "complexity": complexity,
                "resolved_status": resolved_status
            })

        logging.info(f"{num_samples} conversas simuladas geradas.")
        return pd.DataFrame(chat_data)

    @staticmethod
    def enrich_and_save_chat_data(csv_path="consolidated_chat_data.csv", output_path="enriched_chat_data.csv"):
        # Carregar os dados reais existentes
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            logging.warning(f"Arquivo {csv_path} não encontrado. Um novo arquivo será criado.")
            df = pd.DataFrame()

        # Simular novas conversas e adicionar colunas de dados fictícios
        simulated_data = ConversationSimulator.simulate_conversation(num_samples=70)
        simulated_data["id_chat"] = [str(uuid.uuid4()) for _ in range(len(simulated_data))]
        simulated_data["resposta_data"] = pd.to_datetime("today") - pd.to_timedelta(
            np.random.randint(0, 100, size=len(simulated_data)), unit="D"
        )
        simulated_data["id_manual"] = [random.randint(1, 100) for _ in range(len(simulated_data))]
        simulated_data["c_cpf"] = [fake.unique.cpf() for _ in range(len(simulated_data))]

        # Combinar dados reais e simulados e salvar
        enriched_data = pd.concat([df, simulated_data], ignore_index=True)
        enriched_data.to_csv(output_path, index=False)
        logging.info(f"Dados enriquecidos salvos em {output_path}")
