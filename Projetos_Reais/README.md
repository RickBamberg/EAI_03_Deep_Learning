# Projetos Reais - Deep Learning

## 📌 Sobre

Esta pasta contém **projetos completos e deployados** de Deep Learning aplicados a problemas reais. São aplicações end-to-end prontas para uso, com interface web e deployment.

**Objetivo**: Demonstrar o ciclo completo de um projeto de Deep Learning em produção, desde modelagem até disponibilização para usuários finais.

---

## 🎯 Diferença: Projetos_Estudos vs Projetos_Reais

| Aspecto | Projetos_Estudos | Projetos_Reais |
|---------|------------------|----------------|
| **Escopo** | Problemas clássicos (MNIST) | Aplicações práticas |
| **Complexidade** | Controlada (1 notebook) | Alta (múltiplos arquivos) |
| **Deployment** | Opcional | ✅ **Obrigatório** |
| **Interface** | Notebook | Web app (Flask/Streamlit) |
| **Documentação** | Básica (README) | **Completa (README + AGENT_CONTEXT)** |
| **Público-alvo** | Você (aprendizado) | **Usuários finais** |

---

## 📂 Projetos Disponíveis

### 1️⃣ **ArtClassifier/** 🎨

**Problema**: Classificar estilos artísticos de pinturas

**Tipo**: Classificação Multiclasse de Imagens

**Dataset**: 
- Base de pinturas de diferentes estilos
- Categorias: Impressionismo, Cubismo, Realismo, etc.
- Imagens de tamanhos variados

**Arquitetura**: Transfer Learning com modelo pré-treinado
```python
# Base: VGG16, ResNet50, ou MobileNet
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

**Técnicas Aplicadas**:
- Transfer Learning (aproveitamento de modelos ImageNet)
- Data Augmentation (rotação, zoom, flip)
- Fine-tuning (descongelar camadas superiores)

**Deployment**:
- **Interface**: Flask Web App
- **Funcionalidade**: Upload de imagem → Predição de estilo
- **Output**: Classe + Probabilidade + Descrição do estilo

**Estrutura**:
```
ArtClassifier/
├── README.md                  # Documentação do projeto
├── AGENT_CONTEXT.md          # Contexto técnico para IA
├── app.py                    # Aplicação web
├── model/
│   ├── art_classifier.keras  # Modelo treinado
│   └── class_names.pkl       # Nomes das classes
├── notebooks/
│   ├── 01_exploracao.ipynb   # EDA
│   ├── 02_treinamento.ipynb  # Treino do modelo
│   └── 03_avaliacao.ipynb    # Análise de performance
├── static/
│   ├── css/
│   └── images/               # Exemplos de cada estilo
├── templates/
│   └── index.html
├── requirements.txt
└── data/                     # (não incluído - muito grande)
```

**Resultado Esperado**: ~85-90% accuracy

**Aprendizado**:
- Transfer Learning na prática
- Deploy de modelo de imagem
- Interface web para Deep Learning
- Trabalhar com imagens de tamanhos variados

**Documentação**:
- 📄 README.md
- 🤖 AGENT_CONTEXT.md

---

### 2️⃣ **Previsao_Acoes/** 📈

**Problema**: Prever tendência de preços de ações usando dados históricos

**Tipo**: Classificação/Regressão de Séries Temporais

**Dataset**: 
- Dados históricos de ações (Yahoo Finance)
- Features: Open, High, Low, Close, Volume
- Período: 5-10 anos de histórico

**Arquitetura**: LSTM para Séries Temporais
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)  # ou 3 para classificação (Sobe/Neutro/Desce)
])
```

**Técnicas Aplicadas**:
- Feature Engineering (médias móveis, RSI, MACD)
- Janelas deslizantes (sliding windows)
- Normalização de séries temporais
- Walk-forward validation

**Deployment**:
- **Interface**: Flask Web App
- **Funcionalidade**: 
  - Selecionar ação (ticker)
  - Visualizar histórico
  - Ver predição para próximos dias
  - Métricas de confiança
- **Output**: Preço previsto + Probabilidade

**Estrutura**:
```
Previsao_Acoes/
├── README.md
├── AGENT_CONTEXT.md
├── app.py                    # Aplicação Flask
├── model/
│   ├── lstm_stock.keras
│   └── scaler.pkl
├── notebooks/
│   ├── 01_coleta_dados.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_backtesting.ipynb
├── data/
│   ├── raw/                  # Dados brutos
│   └── processed/            # Dados processados
├── templates/
│   └── index.html
├── static/
│   └── css/
│       └── style.css
├── utils/
│   ├── data_loader.py        # Baixar dados do Yahoo Finance
│   ├── features.py           # Criar features
│   └── visualizacao.py       # Gráficos
├── requirements.txt
└── config.yaml               # Configurações (tickers, período, etc.)
```

**Resultado Esperado**: MAE ~2-5% do valor real

**Aprendizado**:
- LSTM aplicado a séries temporais financeiras
- Feature engineering para finanças
- Validação em séries temporais (não shuffle!)
- Dashboard interativo com Streamlit
- Limitações de predição de mercado

**Documentação**:
- 📄 README.md
- 🤖 AGENT_CONTEXT.md

**⚠️ Disclaimer**: 
> Este projeto é **educacional**. Não use para decisões financeiras reais. Mercados são imprevisíveis e modelos não garantem lucro.

---

## 📊 Comparação dos Projetos

| Aspecto | ArtClassifier | Previsao_Acoes |
|---------|---------------|----------------|
| **Tipo** | Classificação de Imagens | Séries Temporais |
| **Arquitetura** | Transfer Learning (VGG16/ResNet) | LSTM |
| **Dataset** | Imagens de arte | Dados financeiros |
| **Deploy** | Flask (upload de imagem) | Flask (seleção de ticker) |
| **Complexidade** | ⭐⭐⭐ Média | ⭐⭐⭐⭐ Alta |
| **Accuracy** | ~85-90% | MAE ~3% |
| **Aprendizado Principal** | Transfer Learning | Séries Temporais |

---

## 🚀 Como Executar os Projetos

### ArtClassifier

```bash
# 1. Navegar para pasta
cd ArtClassifier/

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Rodar aplicação Flask
python app.py

# 5. Acessar
# http://localhost:5000
```

### Previsao_Acoes

```bash
# 1. Navegar para pasta
cd Previsao_Acoes/

# 2. Setup ambiente
python -m venv venv
source venv/bin/activate

# 3. Instalar
pip install -r requirements.txt

# 4. (Opcional) Coletar dados
python utils/data_loader.py --ticker AAPL --years 10

# 5. Rodar aplicação Flask
python app.py

# 6. Acessar
# http://localhost:5000
```

---

## 🎯 Funcionalidades dos Apps

### ArtClassifier - Interface Web

**Fluxo do Usuário**:
1. Acessa URL (localhost ou deploy)
2. Faz upload de imagem de uma pintura
3. Clica em "Classificar"
4. Recebe:
   - Estilo previsto (ex: "Impressionismo")
   - Probabilidade (ex: 87%)
   - Descrição do estilo
   - Artistas famosos desse estilo

**Features**:
- Upload de imagem (JPG, PNG)
- Pré-visualização da imagem
- Processamento em tempo real
- Gráfico de probabilidades por classe
- Exemplos de cada estilo

**Tecnologias**:
- Backend: Flask
- Frontend: HTML/CSS com Jinja2
- Modelo: Keras/TensorFlow (MobileNetV2)

---

### Previsao_Acoes - Dashboard

**Fluxo do Usuário**:
1. Acessa dashboard
2. Seleciona ticker (ex: AAPL, GOOGL)
3. Define período de análise
4. Visualiza:
   - Gráfico histórico
   - Predição para próximos 7 dias
   - Métricas de performance do modelo
   - Indicadores técnicos

**Features**:
- Seleção de ação (dropdown)
- Gráficos interativos (Plotly)
- Atualização de dados em tempo real (opcional)
- Indicadores técnicos (SMA, EMA, RSI)
- Download de dados/predições

**Tecnologias**:
- Framework: Flask
- Dados: yfinance (Yahoo Finance)
- Modelo: Keras/TensorFlow LSTM
- Frontend: HTML/CSS com Jinja2

---

## 📈 Performance Esperada

### ArtClassifier

**Métricas**:
```
Overall Accuracy: 85-90%

Por Classe:
Impressionismo:  Precision 0.88, Recall 0.85
Cubismo:         Precision 0.90, Recall 0.87
Realismo:        Precision 0.83, Recall 0.86
Surrealismo:     Precision 0.85, Recall 0.84
```

**Matriz de Confusão**: Impressionismo às vezes confundido com Realismo

---

### Previsao_Acoes

**Métricas**:
```
MAE (Mean Absolute Error): 2-5% do valor real
RMSE: 3-7%
Direction Accuracy: 55-60% (acima do acaso 50%)
```

**Importante**: 
- Performance varia por ação
- Ações voláteis são mais difíceis
- Modelo funciona melhor para tendências que para volatilidade

---

## 🔧 Tecnologias Utilizadas

### Comuns aos 2 Projetos

| Categoria | Tecnologia |
|-----------|------------|
| **Deep Learning** | TensorFlow, Keras |
| **Data** | NumPy, Pandas |
| **Visualização** | Matplotlib, Seaborn, Plotly |
| **Web** | Flask |
| **Deployment** | Heroku, Railway, Render |

### Específicas por Projeto

**ArtClassifier**:
- PIL/OpenCV (processamento de imagem)
- MobileNetV2 (transfer learning)

**Previsao_Acoes**:
- yfinance (dados de ações)
- MinMaxScaler (normalização)
- scikit-learn (preprocessing)

---

## 💻 Deployment

### Opções de Deploy

#### 1. Heroku (Flask Apps)
```bash
# 1. Criar Procfile
echo "web: python app.py" > Procfile

# 2. Deploy
heroku create myapp
git push heroku main
```

#### 2. Railway
```bash
# 1. Conectar GitHub repo
# 2. Configurar variáveis de ambiente
# 3. Deploy automático
```

---

## 📚 Estrutura de Documentação

Cada projeto tem 2 arquivos de documentação:

### README.md (Humanos)
- Descrição do projeto
- Como executar localmente
- Como usar a aplicação
- Capturas de tela
- Estrutura do projeto
- Melhorias futuras

### AGENT_CONTEXT.md (IA/Técnico)
- Arquitetura detalhada do modelo
- Pipeline de dados completo
- Código-chave comentado
- Hiperparâmetros otimizados
- Métricas detalhadas
- Troubleshooting técnico
- FAQ técnico

---

## 🎓 O Que Você Aprenderá

### Com ArtClassifier
- ✅ Transfer Learning na prática
- ✅ Deploy de modelo CNN
- ✅ Processar uploads de usuário
- ✅ Interface web para DL
- ✅ Lidar com imagens de tamanhos variados

### Com Previsao_Acoes
- ✅ LSTM para séries temporais
- ✅ Feature engineering financeiro
- ✅ Validação temporal (não shuffle)
- ✅ Dashboard interativo
- ✅ Limitações de predição de mercado

### Geral (Ambos)
- ✅ Ciclo completo: Dados → Modelo → Deploy
- ✅ Estruturação de projeto profissional
- ✅ Documentação técnica e não-técnica
- ✅ Deployment em produção
- ✅ Interface para usuários finais

---

## ⚠️ Limitações e Considerações

### ArtClassifier
- Performance depende da qualidade do dataset
- Estilos similares podem ser confundidos
- Funciona melhor com pinturas "típicas" do estilo

### Previsao_Acoes
- **NÃO use para decisões financeiras reais**
- Mercados são influenciados por fatores externos não capturados
- Performance passada ≠ performance futura
- Melhor para análise educacional que trading real

---

## 🚀 Próximos Passos

Após completar estes projetos:

1. **Criar seu próprio projeto deployado**
   - Escolha um problema real
   - Desenvolva end-to-end
   - Faça deploy

2. **Contribuir para projetos open source**
   - Melhorar documentação
   - Adicionar features
   - Reportar bugs

3. **Participar de competições Kaggle**
   - Aplicar técnicas aprendidas
   - Aprender com outros

4. **Portfólio**
   - Adicionar projetos ao GitHub
   - Escrever blog posts sobre o processo
   - Compartilhar no LinkedIn

---

## 🔗 Links Úteis

**Deploy**:
- [Heroku](https://www.heroku.com/)
- [Railway](https://railway.app/)
- [Render](https://render.com/)

**Datasets**:
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers With Code](https://paperswithcode.com/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

**Inspiração**:
- [Awesome Deep Learning Projects](https://github.com/ahmadchan/Awesome-Deep-Learning-Projects)
- [Made With ML](https://madewithml.com/)

---

## 📝 Checklist de Projeto Completo

Use este checklist para seus próprios projetos:

**Desenvolvimento**:
- [ ] Problema bem definido
- [ ] Dataset adequado
- [ ] EDA completa
- [ ] Modelo treinado e validado
- [ ] Performance aceitável

**Código**:
- [ ] Código organizado em módulos
- [ ] Comentários e docstrings
- [ ] requirements.txt
- [ ] .gitignore configurado

**Deployment**:
- [ ] Interface funcional
- [ ] Testes locais OK
- [ ] Deploy em produção
- [ ] URL acessível

**Documentação**:
- [ ] README.md completo
- [ ] AGENT_CONTEXT.md técnico
- [ ] Capturas de tela
- [ ] Instruções de uso

---

**Lembre-se**: Projeto deployado > Projeto no notebook! A capacidade de colocar modelos em produção é o que diferencia cientistas de dados sênior.

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_03*
