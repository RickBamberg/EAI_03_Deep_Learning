"""
🌐 Classificador de Estilos Artísticos - Backend Flask

Este script implementa uma aplicação web simples usando Flask que permite ao usuário enviar
uma imagem de pintura e obter como resposta o estilo artístico previsto pelo modelo treinado.

O modelo é carregado com TensorFlow, e o frontend usa HTML/CSS com renderização via Jinja2.

"""

# 📦 Importação de pacotes necessários

import os
import json
import numpy as np
from PIL import Image # Pillow para manipulação de imagens

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from flask import Flask, request, render_template, redirect, url_for

# --- Inicialização da Aplicação Flask ---
app = Flask(__name__)

# 📁 Configuração do caminho de upload de imagens
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Garante que a pasta de uploads exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🖼️ Tamanho esperado das imagens (deve bater com o que o modelo espera)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 📥 Carregar o modelo e os rótulos (executado uma única vez ao iniciar o app)
print("Carregando o modelo treinado e os rótulos...")

try:
    # Ajusta o caminho para apontar para a pasta 'models'
    model_path = os.path.join('models', 'art_style_classifier_best.keras')
    labels_path = os.path.join('models', 'class_names.json')

    model = load_model(model_path)
    with open(labels_path, 'r') as f:
        class_names = json.load(f)
    print("Modelo e rótulos carregados com sucesso!")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o modelo ou os rótulos. {e}")
    model = None
    class_names = []

# 🧪 Função para carregar e pré-processar a imagem
def preprocess_image(image_path):
    """
    Carrega uma imagem, redimensiona e a pré-processa para o modelo.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        img_array = np.array(img)
        
        # Expande as dimensões para criar um "batch" de 1 imagem
        # O formato passa de (224, 224, 3) para (1, 224, 224, 3)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        return img_array_expanded

    except Exception as e:
        print(f"Erro ao pré-processar a imagem: {e}")
        return None

# 🔗 Rota principal (index)
@app.route('/', methods=['GET'])
def index():
    """Renderiza a página inicial de upload."""
    return render_template('index.html')

# 🔍 Rota de previsão
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe a imagem, processa, faz a previsão e mostra o resultado.
    """
    if model is None:
        return "Erro: O modelo não está carregado. Verifique o console do servidor."

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
   
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Pré-processa a imagem
        processed_image = preprocess_image(filepath)
        
        if processed_image is not None:
            prediction_scores = model.predict(processed_image)
            
            predicted_index = np.argmax(prediction_scores, axis=1)[0]
            
            predicted_style = class_names[predicted_index]
            
            return render_template('result.html', 
                                   prediction=predicted_style, 
                                   image_path=filepath)
        else:
            return "Erro ao processar a imagem."

    return redirect(url_for('index'))

# ▶️ Executa o servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

