# 🎨 Classificador de Estilos Artísticos com Deep Learning

Sistema de classificação de pinturas por estilo artístico usando Transfer Learning com MobileNetV2 e interface web Flask.

---

## 🎯 Objetivo

Criar um modelo de Deep Learning capaz de reconhecer o estilo artístico de uma pintura baseado apenas em sua aparência visual, demonstrando:
- **Transfer Learning** com MobileNetV2 (ImageNet)
- **Data Augmentation** para melhorar generalização
- **Análise de erros** para entender comportamento do modelo
- **Deployment** com Flask para uso interativo

**Resultado**: ~65% accuracy (bom para 7 classes subjetivas e complexas)

---

## 🧠 Como Funciona

O sistema classifica pinturas em estilos artísticos baseado em **padrões visuais** (cores, pinceladas, composição).

### Pipeline
```
Upload de Imagem → Pré-processamento → MobileNetV2 → Classificador → Estilo Previsto
   (usuário)          (224×224)         (congelada)   (customizado)   (+confiança)
```

### Diferencial: Transfer Learning

O modelo **não treina do zero**. Ele aproveita **MobileNetV2** pré-treinada em ImageNet (1.4M imagens):

1. **Base congelada**: Extratora de features já treinada
2. **Classificador customizado**: Apenas camadas finais treinadas
3. **Vantagem**: Rápido, eficiente, menos dados necessários

**Resultado**: Reduz de milhões para ~500k parâmetros treináveis!

---

## 🏗️ Arquitetura do Modelo

### MobileNetV2 + Classificador Customizado

```python
Input (224, 224, 3)
    ↓
Pré-processamento MobileNetV2
    ↓
MobileNetV2 Base (congelada - ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense (256 units, ReLU)
    ↓
Dropout (50%)
    ↓
Dense (7 units, Softmax)
```

**Por que MobileNetV2?**
- ✅ Leve (~14 MB vs ~90 MB ResNet50)
- ✅ Rápida inferência (ideal para CPU)
- ✅ Boa performance em classificação
- ✅ Perfeita para deployment Flask

**Parâmetros**: ~2.5M total, ~500k treináveis (apenas classificador)

---

## 🖼️ Dataset - WikiArt

### Fonte
- **Nome**: WikiArt - Painter by Numbers
- **URL**: https://www.kaggle.com/datasets/ipythonx/painter-by-numbers
- **Estrutura**: Pastas por estilo artístico

### 7 Estilos Artísticos

```
1. Abstract_Expressionism    (Expressionismo Abstrato)
2. Analytical_Cubism         (Cubismo Analítico)
3. Art_Nouveau_Modern        (Art Nouveau Moderno)
4. Baroque                   (Barroco)
5. Cubism                    (Cubismo)
6. Expressionism             (Expressionismo)
7. Impressionism             (Impressionismo)
```

### Organização
```
WikiArt/
├── Abstract_Expressionism/
│   ├── painting1.jpg
│   ├── painting2.jpg
│   └── ...
├── Baroque/
│   └── ...
└── Impressionism/
    └── ...
```

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/RickBamberg/ArtClassifier.git
cd ArtClassifier

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar Aplicação Flask

```bash
python app.py
```

**Acesse**: http://localhost:5000

### 3. Usar Interface

1. **Upload** de imagem de uma pintura (JPG, PNG)
2. Clique em **"Classificar"**
3. Aguarde processamento (~2 segundos)
4. Veja resultado:
   - **Estilo previsto** (ex: "Impressionism")
   - **Confiança** (ex: 87%)
   - **Descrição** do estilo (opcional)

---

## 📁 Estrutura do Projeto

```
ArtClassifier/
├── app.py                      # 🌐 Backend Flask
├── requirements.txt            # 📦 Dependências
├── README.md                   # 📄 Este arquivo
├── AGENT_CONTEXT.md           # 🤖 Documentação técnica
│
├── models/                     # 💾 Modelo treinado
│   ├── art_style_classifier_best.keras
│   └── class_names.json
│
├── notebooks/
│   └── ArtClassifier.ipynb    # 📓 Treinamento e análise
│
├── static/
│   └── uploads/               # 📸 Imagens enviadas
│
└── templates/                  # 🖼️ Interface web
    ├── index.html
    └── result.html
```

---

## 🧪 Treinamento do Modelo

### Técnicas Aplicadas

**1. Transfer Learning**:
```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False
)
base_model.trainable = False  # Congelar
```

**2. Data Augmentation**:
```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
```

**3. Callbacks**:
- **EarlyStopping**: Para quando validação piora
- **ModelCheckpoint**: Salva melhor modelo
- **ReduceLROnPlateau**: Reduz learning rate se estagnado

### Resultado Final

**Validation Accuracy**: ~65%

**Interpretação**: 
- ✅ Bom para 7 classes subjetivas e visualmente similares
- ✅ Tarefa é complexa até para humanos
- ✅ Sem contexto histórico (apenas visual)

---

## 📊 Análise de Resultados

### Matriz de Confusão - Principais Erros

```
Baroque vs Impressionism:        19 erros
Expressionism vs Abstract_Expr:  12 erros
Cubism vs Analytical_Cubism:      8 erros
```

**Por quê?**
- **Baroque ↔ Impressionism**: Ambos usam céus luminosos
- **Expressionism ↔ Abstract**: Transição gradual de estilos
- **Cubism ↔ Analytical**: Variações do mesmo movimento

### Case Study: Generalização do Modelo

**Pintura**: "View of Scheveningen Sands" (Barroco, 1641)  
**Predição**: Impressionismo (ERRADO)  
**Confiança**: Alta

**Análise**:
```
Características visuais:
✓ Céu vasto e luminoso
✓ Cores claras e atmosféricas
✓ Pinceladas suaves

Conclusão do modelo:
"Céus luminosos + pinceladas suaves" = Impressionismo
```

**Interpretação**: 
> ✅ **Modelo está GENERALIZANDO**, não memorizando!  
> O erro é compreensível: baseado apenas em padrões visuais, sem contexto histórico.

Este comportamento é **positivo** - mostra que o modelo aprendeu features visuais reais.

---

## 🌐 Aplicação Flask

### Backend (app.py)

```python
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Carregar modelo e classes
model = load_model('models/art_style_classifier_best.keras')
with open('models/class_names.json') as f:
    class_names = json.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    # Processar imagem
    img = preprocess_image(file)
    
    # Predição
    prediction = model.predict(img)
    predicted_index = prediction.argmax()
    predicted_style = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])
    
    return render_template('result.html',
                         prediction=predicted_style,
                         confidence=f"{confidence*100:.1f}%",
                         image_path=file_path)
```

### Frontend

**index.html**: Formulário de upload  
**result.html**: Exibição de resultado com imagem

---

## 📚 Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
|-----------|-----------|-----|
| **Deep Learning** | TensorFlow/Keras | Modelo e Transfer Learning |
| **Modelo Base** | MobileNetV2 | Extração de features |
| **Pré-processamento** | PIL (Pillow) | Manipulação de imagens |
| **Web** | Flask | Backend |
| **Frontend** | HTML/CSS | Interface |
| **Dados** | NumPy | Arrays e operações |

---

## 📊 Performance e Limitações

### Quando o Modelo Funciona Bem

- ✅ Pinturas "típicas" de cada estilo
- ✅ Estilos bem distintos (Cubismo vs Impressionismo)
- ✅ Imagens de boa qualidade

### Quando o Modelo Falha

- ❌ Estilos muito similares (Cubismo vs Cubismo Analítico)
- ❌ Pinturas de transição entre estilos
- ❌ Obras muito atípicas do estilo
- ❌ Sem contexto histórico (ano, artista)

### Limitações Técnicas

**O que o modelo VÊ**:
- Cores e paleta
- Padrões de pinceladas
- Composição visual
- Texturas

**O que o modelo NÃO VÊ**:
- Contexto histórico
- Período temporal
- Artista específico
- Significado cultural

---

## 🔮 Melhorias Futuras

### Modelo
- [ ] Fine-tuning: Descongelar camadas superiores
- [ ] Ensemble: MobileNetV2 + EfficientNet + ResNet
- [ ] Aumentar dataset (mais estilos, mais imagens)
- [ ] Adicionar metadata (artista, período, técnica)

### Aplicação
- [ ] Deploy em cloud (Heroku, Render, Railway)
- [ ] Top-3 predições com confiança
- [ ] Galeria de exemplos de cada estilo
- [ ] Upload múltiplo de imagens
- [ ] Comparação lado a lado

### Análise
- [ ] Grad-CAM para visualizar features aprendidas
- [ ] Análise por artista específico
- [ ] Clustering de estilos similares
- [ ] Explicabilidade (LIME, SHAP)

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

**Como contribuir**:
1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

**Ideias de contribuição**:
- Adicionar mais estilos artísticos
- Implementar fine-tuning
- Melhorar interface web
- Adicionar testes automatizados

---

## 📖 Recursos Adicionais

### Transfer Learning
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

### WikiArt Dataset
- [Kaggle Dataset](https://www.kaggle.com/datasets/ipythonx/painter-by-numbers)
- [WikiArt Website](https://www.wikiart.org/)

### Flask
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)

---

## 📝 Citação

Se usar este projeto, por favor cite:

```
@misc{art_classifier_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Classificador de Estilos Artísticos com Transfer Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RickBamberg/ArtClassifier}
}
```

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 🙏 Agradecimentos

- [WikiArt](https://www.wikiart.org/) - Dataset de pinturas
- [TensorFlow](https://www.tensorflow.org/) - Framework de Deep Learning
- [Kaggle](https://www.kaggle.com/) - Plataforma de datasets
- Comunidade de arte e tecnologia

---

**💡 Lembre-se**: O modelo generaliza baseado em padrões visuais. Erros são naturais e instrutivos!

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_03*
