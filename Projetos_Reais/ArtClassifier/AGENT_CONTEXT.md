# AGENT_CONTEXT.md - ArtClassifier (Classificador de Estilos Artísticos)

> **Propósito**: Contexto técnico completo do projeto ArtClassifier deployado  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Projeto real com Transfer Learning e deployment Flask

## RESUMO EXECUTIVO

**Objetivo**: Classificar pinturas em estilos artísticos usando Transfer Learning  
**Arquitetura**: MobileNetV2 (ImageNet) + Fine-tuning  
**Dataset**: WikiArt - Painter by Numbers (6-7 estilos)  
**Resultado**: ~65% accuracy (esperado para problema complexo)  
**Deployment**: Flask web app com upload de imagens  
**Diferencial**: Análise profunda de erros e generalização

---

## DATASET - WIKIART PAINTER BY NUMBERS

### Fonte
- **URL**: https://www.kaggle.com/datasets/ipythonx/painter-by-numbers
- **Estrutura**: Pastas por estilo artístico
- **Estilos**: 7 classes

### Classes (Estilos Artísticos)

```
1. Abstract_Expressionism    (Expressionismo Abstrato)
2. Analytical_Cubism         (Cubismo Analítico)
3. Art_Nouveau_Modern        (Art Nouveau Moderno)
4. Baroque                   (Barroco)
5. Cubism                    (Cubismo)
6. Expressionism             (Expressionismo)
7. Impressionism             (Impressionismo)
```

### Estrutura de Diretórios
```
WikiArt/
├── Abstract_Expressionism/
│   ├── painting1.jpg
│   ├── painting2.jpg
│   └── ...
├── Analytical_Cubism/
│   └── ...
├── Baroque/
│   └── ...
└── ...
```

### Pré-processamento de Nomes

**Problema**: Caracteres especiais, acentos, espaços  
**Solução**: Função `sanitize_filename()`

```python
import unicodedata
import re

def sanitize_filename(filename):
    """
    Limpa nome de arquivo removendo caracteres especiais e acentos
    
    Exemplo:
        "abc.-nome(1) ç.jpg" -> "abc_-nome_1__c.jpg"
    """
    # 1. Normalizar e remover acentos
    sanitized = unicodedata.normalize('NFKD', filename)
    sanitized = sanitized.encode('ascii', 'ignore').decode('ascii')
    
    # 2. Substituir não-alfanuméricos por underscore
    sanitized = re.sub(r'[^\w.\-]', '_', sanitized)
    
    # 3. Evitar underscores múltiplos
    sanitized = re.sub(r'__+', '_', sanitized)
    
    return sanitized

# Aplicar em todas as imagens
DATA_DIR = "path/to/WikiArt/"
renamed_count = 0

for dirpath, dirnames, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        original_filepath = os.path.join(dirpath, filename)
        sanitized_name = sanitize_filename(filename)
        
        if sanitized_name != filename:
            new_filepath = os.path.join(dirpath, sanitized_name)
            
            # Evitar colisões
            counter = 1
            while os.path.exists(new_filepath):
                name, ext = os.path.splitext(sanitized_name)
                new_filepath = os.path.join(dirpath, f"{name}_{counter}{ext}")
                counter += 1
            
            os.rename(original_filepath, new_filepath)
            renamed_count += 1
```

---

## PIPELINE DE TREINAMENTO

### 1. Carregamento de Dados

```python
import tensorflow as tf

# Parâmetros
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Carregar dataset com split automático
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Obter nomes das classes
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# Salvar nomes para uso no Flask
import json
with open('models/class_names.json', 'w') as f:
    json.dump(class_names, f)
```

### 2. Data Augmentation

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),  # ±10%
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# Aplicar no pipeline
def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_ds_augmented = train_ds.map(augment)
```

### 3. Otimização de Performance

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

---

## ARQUITETURA DO MODELO - TRANSFER LEARNING

### MobileNetV2 Base

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Carregar MobileNetV2 pré-treinada (ImageNet)
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,  # Remove classificador original
    weights='imagenet'
)

# Congelar camadas da base (Transfer Learning)
base_model.trainable = False
```

**Por Que MobileNetV2?**:
- Leve (~14 MB)
- Rápida inferência
- Boa performance em classificação
- Ideal para deployment

### Arquitetura Completa

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout
)

# Input
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Pré-processamento MobileNetV2
x = preprocess_input(inputs)

# Base congelada
x = base_model(x, training=False)

# Classificador customizado
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Modelo final
model = Model(inputs, outputs)
```

**Estrutura Resumida**:
```
Input (224, 224, 3)
    ↓
Preprocess (MobileNetV2)
    ↓
MobileNetV2 Base (congelada)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu)
    ↓
Dropout(0.5)
    ↓
Dense(NUM_CLASSES, softmax)
```

### Compilação

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Labels como inteiros
    metrics=['accuracy']
)

model.summary()
```

**Parâmetros**:
```
Total params: ~2.5 million
Trainable params: ~500k (apenas classificador)
Non-trainable params: ~2 million (base congelada)
```

---

## TREINAMENTO

### Callbacks

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

callbacks = [
    # Salvar melhor modelo
    ModelCheckpoint(
        'models/art_style_classifier_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Parar se não melhorar
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduzir LR se estagnado
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
```

### Execução

```python
EPOCHS = 50

history = model.fit(
    train_ds_augmented,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
```

---

## RESULTADOS

### Performance Final

```
Validation Accuracy: ~65%
Validation Loss: ~1.2
```

**Interpretação**:
- 65% é **bom** para 7 classes de estilos artísticos
- Tarefa é **subjetiva** (até humanos divergem)
- Estilos têm sobreposição visual (Cubismo vs Cubismo Analítico)

### Curvas de Aprendizado

```python
import matplotlib.pyplot as plt

# Plotar loss e accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Validation')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Validation')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.show()
```

---

## ANÁLISE DE ERROS

### Matriz de Confusão

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predições no conjunto de validação
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Plotar
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

### Confusões Mais Comuns

```
Baroque vs Impressionism:        19 erros
Expressionism vs Abstract_Expr:  12 erros
Cubism vs Analytical_Cubism:      8 erros
```

**Por Quê?**:
- **Baroque vs Impressionism**: Ambos usam céus vastos e luminosos
- **Expressionism vs Abstract**: Transição gradual entre estilos
- **Cubism vs Analytical**: Variações do mesmo movimento

---

## INSIGHTS - GENERALIZAÇÃO DO MODELO

### Caso de Estudo: "View of Scheveningen Sands"

**Pintura**: Barroca (Hendrick van Anthonissen, 1641)  
**Predição do Modelo**: Impressionismo  
**Confiança**: Alta

**Análise**:
```
Características da pintura:
- Céu vasto e luminoso
- Cores claras e atmosféricas
- Pinceladas suaves

Por que o modelo errou?
- Aprendeu que "céus luminosos + pinceladas suaves" = Impressionismo
- Não tem contexto histórico (data, artista)
- Decisão baseada APENAS em features visuais
```

**Conclusão**: ✅ **Modelo está generalizando**, não apenas memorizando!

Este erro é **esperado e positivo**:
- Mostra que modelo aprendeu padrões visuais
- Demonstra capacidade de generalização
- Limitação inerente: Sem contexto temporal/histórico

---

## DEPLOYMENT - FLASK WEB APP

### Estrutura de Arquivos

```
art_classifier/
├── app.py
├── models/
│   ├── art_style_classifier_best.keras
│   └── class_names.json
├── static/
│   └── uploads/
├── templates/
│   ├── index.html
│   └── result.html
└── requirements.txt
```

### app.py - Backend Flask

```python
import os
import json
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from flask import Flask, request, render_template, redirect, url_for

# Inicialização
app = Flask(__name__)

# Configuração
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Carregar modelo e classes
model_path = os.path.join('models', 'art_style_classifier_best.keras')
labels_path = os.path.join('models', 'class_names.json')

model = load_model(model_path)
with open(labels_path, 'r') as f:
    class_names = json.load(f)

# Pré-processamento
def preprocess_image(image_path):
    """Carrega, redimensiona e pré-processa imagem"""
    try:
        # Carregar e converter para RGB
        img = Image.open(image_path).convert('RGB')
        
        # Redimensionar
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Para array numpy
        img_array = np.array(img)
        
        # Expandir dimensões (batch de 1)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        return img_array_expanded
    except Exception as e:
        print(f"Erro ao pré-processar: {e}")
        return None

# Rotas
@app.route('/', methods=['GET'])
def index():
    """Página inicial com formulário de upload"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recebe imagem, faz predição e retorna resultado"""
    if model is None:
        return "Erro: Modelo não carregado"
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        # Salvar arquivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Pré-processar
        processed_image = preprocess_image(filepath)
        
        if processed_image is not None:
            # Predição
            prediction_scores = model.predict(processed_image)
            predicted_index = np.argmax(prediction_scores, axis=1)[0]
            predicted_style = class_names[predicted_index]
            confidence = float(prediction_scores[0][predicted_index])
            
            return render_template(
                'result.html',
                prediction=predicted_style,
                confidence=f"{confidence*100:.1f}%",
                image_path=filepath
            )
        else:
            return "Erro ao processar imagem"
    
    return redirect(url_for('index'))

# Executar
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### templates/index.html

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>🎨 Classificador de Estilos Artísticos</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            text-align: center;
        }
        h1 { color: #333; }
        form { margin: 30px 0; }
        input[type="file"] {
            padding: 10px;
            margin: 20px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <h1>🎨 Classificador de Estilos Artísticos</h1>
    <p>Envie uma imagem de uma pintura e descubra seu estilo artístico!</p>
    
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br>
        <button type="submit">Classificar</button>
    </form>
</body>
</html>
```

### templates/result.html

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Resultado</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            text-align: center;
        }
        img {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 10px;
            margin: 20px 0;
        }
        .result {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        a {
            color: #4CAF50;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <h1>Resultado da Classificação</h1>
    
    <img src="{{ image_path }}" alt="Imagem enviada">
    
    <div class="result">
        <h2>Estilo Predito: <strong>{{ prediction }}</strong></h2>
        <p>Confiança: {{ confidence }}</p>
    </div>
    
    <a href="/">← Classificar outra imagem</a>
</body>
</html>
```

---

## COMO EXECUTAR

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/usuario/art-classifier.git
cd art-classifier

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. requirements.txt

```txt
tensorflow>=2.10.0
flask>=2.0.0
pillow>=9.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### 3. Executar Flask

```bash
python app.py
```

**Acesse**: http://localhost:5000

---

## MELHORIAS FUTURAS

### Modelo
- [ ] Fine-tuning: Descongelar camadas superiores da MobileNetV2
- [ ] Ensemble: Combinar MobileNetV2 + EfficientNet + ResNet
- [ ] Aumentar dataset (mais estilos, mais imagens por estilo)
- [ ] Adicionar metadata (artista, período, técnica)

### Aplicação
- [ ] Deploy em cloud (Heroku, Render, Railway)
- [ ] Adicionar confiança das top-3 predições
- [ ] Mostrar exemplos de cada estilo
- [ ] Permitir múltiplas imagens simultaneamente
- [ ] API REST para integração

### Análise
- [ ] Grad-CAM para visualizar features aprendidas
- [ ] Análise por artista específico
- [ ] Clustering de estilos similares

---

## FAQ TÉCNICO

**Q: Por que apenas 65% accuracy?**
A: Tarefa é subjetiva e complexa. Estilos têm sobreposição visual. Sem contexto histórico, decisão é apenas visual. 65% é competitivo para este problema.

**Q: Por que MobileNetV2 em vez de ResNet/VGG?**
A: MobileNetV2 é leve (~14 MB vs ~90 MB ResNet50), rápida, e suficiente para deployment. Ideal para Flask em CPU.

**Q: Como melhorar accuracy?**
A: (1) Fine-tuning (descongelar base), (2) Mais dados, (3) Ensemble de modelos, (4) Metadata (artista, período).

**Q: O que é preprocess_input?**
A: Função da MobileNetV2 que normaliza imagens para range esperado pelo modelo. Converte RGB [0-255] para [-1, 1].

**Q: Por que GlobalAveragePooling em vez de Flatten?**
A: GAP reduz dimensionalidade drasticamente (1280 valores em vez de 7×7×1280=62720), evitando overfit e reduzindo parâmetros.

**Q: Como fazer fine-tuning?**
```python
# Após treino inicial
base_model.trainable = True

# Congelar apenas primeiras camadas
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompilar com LR menor
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar novamente
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=10)
```

---

## TAGS DE BUSCA

`#transfer-learning` `#mobilenetv2` `#classificacao-imagens` `#estilos-artisticos` `#flask` `#deployment` `#data-augmentation` `#fine-tuning` `#wikiart` `#keras` `#tensorflow` `#computer-vision`

---

**Versão**: 1.0  
**Compatibilidade**: TensorFlow 2.x, Flask 2.x  
**Uso recomendado**: Classificação de arte, educação sobre Transfer Learning, exemplo de deployment
