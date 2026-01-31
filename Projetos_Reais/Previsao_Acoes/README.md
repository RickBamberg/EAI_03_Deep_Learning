# 📈 Previsão de Preços de Ações com LSTM

Sistema inteligente de previsão de preços de ações da B3 usando redes neurais LSTM (Long Short-Term Memory) com atualização automática e interface web Flask.

---

## 🎯 Objetivo

Demonstrar aplicação prática de Deep Learning em séries temporais financeiras, implementando:
- Rede LSTM para capturar padrões temporais
- Sistema de atualização automática com **fine-tuning**
- Interface web amigável para usuários finais
- Boas práticas de Machine Learning em produção

**⚠️ IMPORTANTE**: Este projeto é **educacional**. NÃO use para decisões financeiras reais.

---

## 🧠 Como Funciona

O sistema prevê o **próximo preço de fechamento ajustado** de uma ação baseado nos **últimos 60 dias** de histórico.

### Pipeline
```
Yahoo Finance → Normalização → LSTM → Predição → Interface Web
   (dados)      (0 a 1)      (2 layers)  (próximo dia)   (Flask)
```

### Diferencial: Sistema Inteligente

O sistema **não re-treina do zero** toda vez. Ele:

1. **Primeira vez**: Treina modelo com dados históricos (2018-hoje)
2. **Uso diário**: Carrega modelo existente (rápido)
3. **Desatualizado**: Faz **fine-tuning** apenas com dados novos

**Resultado**: Predições sempre atualizadas sem overhead de re-treinamento completo!

---

## 🏗️ Arquitetura do Modelo

### LSTM de 2 Camadas

```python
Input (60 dias, 1 feature)
    ↓
LSTM Layer 1 (50 units) + Dropout (20%)
    ↓
LSTM Layer 2 (50 units) + Dropout (20%)
    ↓
Dense (1 unit - preço previsto)
```

**Por que LSTM?**
- ✅ Captura dependências temporais de longo prazo
- ✅ Memória celular para padrões históricos
- ✅ Gates controlam informação relevante
- ✅ Superior a RNN simples (evita vanishing gradient)

**Parâmetros**: ~15.000 treináveis

---

## 📊 Dataset

### Fonte: Yahoo Finance (yfinance)

```python
ticker = 'PETR4.SA'  # Petrobras
dados = yf.download(ticker, start='2018-01-01', end='hoje', auto_adjust=True)
```

**Auto-adjust**: Ajusta preços por dividendos e splits automaticamente

### Empresas Suportadas

Qualquer ação da B3 com sufixo `.SA`:
- `PETR4.SA` - Petrobras
- `VALE3.SA` - Vale
- `ITUB4.SA` - Itaú
- `BBDC4.SA` - Bradesco
- `MGLU3.SA` - Magazine Luiza
- E centenas de outras...

Lista completa em: `data/Tickers_B3.csv`

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/RickBamberg/Previsao_Acoes.git
cd Previsao_Acoes

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar Aplicação

```bash
python app.py
```

**Acesse**: http://localhost:5000

### 3. Usar Interface

1. Selecione ou digite o ticker (ex: `PETR4.SA`)
2. Clique em **"Prever Preço"**
3. Aguarde processamento (primeira vez pode demorar ~2 min)
4. Veja resultado: **R$ XX.XX**

**Dica**: Na primeira vez para cada ação, o modelo será treinado. Usos seguintes são instantâneos!

---

## 📁 Estrutura do Projeto

```
Previsao_Acoes/
├── app.py                      # 🌐 Backend Flask
├── requirements.txt            # 📦 Dependências
├── README.md                   # 📄 Este arquivo
├── AGENT_CONTEXT.md           # 🤖 Documentação técnica
│
├── data/
│   └── Tickers_B3.csv         # 📋 Lista de empresas B3
│
├── models/                     # 💾 Modelos treinados (gerados automaticamente)
│   ├── petr4.sa_model.keras
│   ├── petr4.sa_scaler.pkl
│   └── ... (um modelo por ação)
│
├── notebook/
│   └── Previsao_Acoes.ipynb   # 📓 Versão acadêmica
│
├── static/
│   └── css/
│       └── style.css          # 🎨 Estilos
│
└── templates/
    └── index.html             # 🖼️ Interface web
```

---

## 🧪 Versão Notebook (Acadêmica)

O notebook `Previsao_Acoes.ipynb` demonstra passo a passo:

1. ✅ Coleta de dados com `yfinance`
2. ✅ Pré-processamento e normalização
3. ✅ Criação de sequências (sliding window)
4. ✅ Construção do modelo LSTM
5. ✅ Treinamento e visualização de erro
6. ✅ Previsão e interpretação

**Uso**: Executar células sequencialmente para entender o processo completo.

---

## 🌐 Versão Flask (Produção)

A aplicação web oferece:

### Funcionalidades

- ✅ **Busca inteligente**: Autocompletar de empresas B3
- ✅ **Atualização automática**: Fine-tuning diário
- ✅ **Cache de modelos**: Predições rápidas
- ✅ **Interface moderna**: Design responsivo
- ✅ **Tratamento de erros**: Mensagens claras

### Fluxo Interno

```python
# 1. Usuário seleciona ação
ticker = 'PETR4.SA'

# 2. Sistema verifica modelo
if modelo_nao_existe:
    treinar_do_zero()
elif modelo_desatualizado:
    fazer_fine_tuning()
else:
    carregar_modelo()

# 3. Fazer predição
preco_previsto = modelo.predict(ultimos_60_dias)

# 4. Mostrar resultado
return f"R$ {preco_previsto:.2f}"
```

---

## 📚 Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
|-----------|-----------|-----|
| **Deep Learning** | TensorFlow/Keras | Modelo LSTM |
| **Dados** | yfinance | Download de cotações |
| **Pré-processamento** | scikit-learn | MinMaxScaler |
| **Manipulação** | Pandas, NumPy | DataFrames e arrays |
| **Visualização** | Matplotlib | Gráficos (notebook) |
| **Web** | Flask | Backend |
| **Frontend** | HTML/CSS | Interface |
| **Persistência** | joblib | Salvar scaler |

---

## 📊 Performance Esperada

### Métricas Típicas

```
MAE (Mean Absolute Error):  2-5% do valor real
Direction Accuracy:         55-60%
```

**Exemplo**:
```
Valor real:     R$ 40.00
Previsto:       R$ 38.50
Erro absoluto:  R$ 1.50 (3.75%) ✅ Bom
```

### Quando o Modelo Funciona Bem

- ✅ Ações líquidas (alto volume)
- ✅ Tendências claras
- ✅ Padrões sazonais
- ✅ Mercado estável

### Quando o Modelo Falha

- ❌ Eventos inesperados (notícias, crises)
- ❌ Ações muito voláteis
- ❌ Baixa liquidez
- ❌ Mudanças estruturais na empresa

---

## ⚠️ Limitações e Disclaimer

### Por Que Previsão de Ações é Difícil?

1. **Mercados são semi-aleatórios** (Efficient Market Hypothesis)
2. **Eventos imprevisíveis** (política, economia, pandemias)
3. **Horizonte curto** (1 dia) é muito ruidoso
4. **Modelo vê apenas preços**, não fundamentos

### O Que o Modelo NÃO Faz

- ❌ Prever crashes ou rallies
- ❌ Considerar notícias
- ❌ Analisar balanços financeiros
- ❌ Garantir lucro

### Uso Responsável

> **DISCLAIMER**: Este projeto é para fins **educacionais**. Não use para decisões de investimento reais. Mercado financeiro envolve riscos. Consulte um profissional certificado (CNPI, CEA) antes de investir.

---

## 🔮 Melhorias Futuras

### Modelo
- [ ] Adicionar features: Volume, RSI, MACD
- [ ] Multi-step prediction (próximos 5 dias)
- [ ] Attention mechanism
- [ ] Ensemble (LSTM + GRU)

### Dados
- [ ] Sentimento de notícias (NLP)
- [ ] Dados macroeconômicos (Selic, dólar)
- [ ] Correlação com índices (Ibovespa)

### Interface
- [ ] Gráfico interativo (Plotly)
- [ ] Intervalo de confiança
- [ ] Histórico de predições vs real
- [ ] Comparar múltiplas ações

### Deploy
- [ ] Containerizar com Docker
- [ ] Deploy em cloud (Heroku, Render, Railway)
- [ ] API REST
- [ ] Autenticação de usuários

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
- Adicionar mais features ao modelo
- Melhorar interface web
- Implementar testes automatizados
- Documentação adicional

---

## 📖 Recursos Adicionais

### Aprender Mais sobre LSTM
- [Understanding LSTM Networks - colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)

### Séries Temporais Financeiras
- [Time Series Forecasting - TensorFlow](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Stock Prediction Kaggle](https://www.kaggle.com/competitions?search=stock)

### yfinance
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Yahoo Finance API Guide](https://aroussi.com/post/python-yahoo-finance)

---

## 📝 Citação

Se usar este projeto, por favor cite:

```
@misc{previsao_acoes_lstm_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Previsão de Ações com LSTM - Sistema Inteligente},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RickBamberg/Previsao_Acoes}
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

- [yfinance](https://github.com/ranaroussi/yfinance) - API de dados financeiros
- [TensorFlow](https://www.tensorflow.org/) - Framework de Deep Learning
- [Flask](https://flask.palletsprojects.com/) - Framework web
- Comunidade B3 e investidores brasileiros

---

**💡 Lembre-se**: Use este projeto para aprender sobre LSTM e séries temporais. Investimentos reais requerem análise muito mais profunda!

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_03*
