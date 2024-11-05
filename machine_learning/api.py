from flask import Flask, request, jsonify
from model_predict import ModelPredict
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

predictor = ModelPredict()


@app.route('/predict/complexity', methods=['POST'])
def predict_complexity():
    """Endpoint para prever o nível de complexidade usando o modelo de regressão."""
    data = request.get_json()
    num_mensagens = data.get('num_mensagens')
    media_palavras = data.get('media_palavras')
    duracao_conversa = data.get('duracao_conversa')

    if None in (num_mensagens, media_palavras, duracao_conversa):
        return jsonify({'error': 'Os campos num_mensagens, media_palavras, e duracao_conversa são obrigatórios'}), 400

    predicted_complexity = predictor.predict_complexity(num_mensagens, media_palavras, duracao_conversa)
    if predicted_complexity is None:
        return jsonify({'error': 'Erro ao fazer a previsão de complexidade'}), 500

    return jsonify({'complexidade_predita': predicted_complexity})


@app.route('/predict/status', methods=['POST'])
def predict_status():
    """Endpoint para prever o status do diagnóstico usando o modelo de classificação."""
    data = request.get_json()
    num_mensagens = data.get('num_mensagens')
    media_palavras = data.get('media_palavras')
    duracao_conversa = data.get('duracao_conversa')
    complexidade_predita = data.get('complexidade_predita')

    if None in (num_mensagens, media_palavras, duracao_conversa, complexidade_predita):
        return jsonify({'error': 'Os campos num_mensagens, media_palavras, duracao_conversa e '
                                 'complexidade_predita são obrigatórios'}), 400

    predicted_status = predictor.predict_status(num_mensagens, media_palavras, duracao_conversa, complexidade_predita)
    if predicted_status is None:
        return jsonify({'error': 'Erro ao fazer a previsão de status de diagnóstico'}), 500

    return jsonify({'status_diagnostico': predicted_status})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
