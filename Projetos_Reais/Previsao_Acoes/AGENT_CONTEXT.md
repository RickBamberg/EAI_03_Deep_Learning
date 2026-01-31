# AGENT_CONTEXT.md - Previsão de Ações com LSTM

> **Propósito**: Contexto técnico completo do sistema de previsão de preços de ações  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Projeto real com LSTM e deployment Flask com atualização automática

## RESUMO EXECUTIVO

**Objetivo**: Prever preço de fechamento ajustado do próximo dia útil de ações da B3  
**Arquitetura**: LSTM de 2 camadas com Dropout  
**Janela Temporal**: 60 dias históricos  
**Fonte de Dados**: Yahoo Finance (yfinance)  
**Resultado**: Previsão em tempo real via Flask  
**Diferencial**: Sistema inteligente de atualização automática com fine-tuning

---

## PROBLEMA - SÉRIES TEMPORAIS FINANCEIRAS

### Características do Problema

**Tipo**: Previsão de séries temporais (time series forecasting)  
**Target**: Preço de fechamento ajustado do dia seguinte  
**Input**: Últimos 60 dias de preços de fechamento  
**Horizonte**: 1 dia à frente (next-day prediction)

### Desafios

1. **Volatilidade**: Mercado financeiro é imprevisível
2. **Eventos externos**: Notícias, política, economia global
3. **Ruído**: Flutuações aleatórias de curto prazo
4. **Não-estacionariedade**: Tendências mudam ao longo do tempo

### Por Que LSTM?

- ✅ Captura dependências temporais de longo prazo
- ✅ Memória celular para padrões históricos
- ✅ Gates controlam fluxo de informação relevante
- ✅ Superior a RNN simples (evita vanishing gradient)

---

## PIPELINE DE DADOS

### 1. Coleta de Dados com yfinance

```python
import yfinance as yf

# Ticker: Código da ação + ".SA" para B3
ticker = 'PETR4.SA'  # Petrobras PN

# Download histórico
dados = yf.download(
    ticker,
    start='2018-01-01',
    end='2024-05-10',
    auto_adjust=True  # Ajusta dividendos e splits
)

# Extrair apenas preço de fechamento
dados_fechamento = dados['Close'].values.reshape(-1, 1)
```

**Auto-adjust**: Importante!
- Ajusta preços por dividendos e splits
- Garante continuidade temporal
- Evita "saltos" artificiais nos dados

### 2. Normalização (MinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
dados_scaled = scaler.fit_transform(dados_fechamento)
```

**Por que normalizar?**:
- LSTM funciona melhor com valores em [0, 1]
- Evita saturação das funções de ativação (tanh, sigmoid)
- Acelera convergência

**CRÍTICO**: 
- Fit no treino, transform no teste
- Salvar scaler para uso em produção

### 3. Criação de Sequências (Sliding Window)

```python
tamanho_sequencia = 60  # Últimos 60 dias

X_train = []
y_train = []

for i in range(tamanho_sequencia, len(dados_scaled)):
    X_train.append(dados_scaled[i-tamanho_sequencia:i, 0])
    y_train.append(dados_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape para LSTM: (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

**Estrutura dos Dados**:
```
X_train.shape: (num_samples, 60, 1)
y_train.shape: (num_samples,)

Exemplo:
X[0] = [dia1, dia2, ..., dia60]  → y[0] = dia61
X[1] = [dia2, dia3, ..., dia61]  → y[1] = dia62
...
```

---

## ARQUITETURA DO MODELO - LSTM

### Estrutura Completa

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

modelo = Sequential([
    Input(shape=(tamanho_sequencia, 1)),  # (60, 1)
    
    # LSTM Layer 1
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    # LSTM Layer 2
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    
    # Output
    Dense(units=1)  # Previsão de valor único
])

modelo.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

modelo.summary()
```

### Detalhamento das Camadas

**LSTM Layer 1** (return_sequences=True):
```python
units=50
- 50 células LSTM
- return_sequences=True → Retorna sequência completa
- Output shape: (batch, 60, 50)
```

**Dropout(0.2)**:
- Desativa 20% dos neurônios aleatoriamente
- Previne overfitting

**LSTM Layer 2** (return_sequences=False):
```python
units=50
- 50 células LSTM
- return_sequences=False → Retorna apenas último timestep
- Output shape: (batch, 50)
```

**Dense(1)**:
- Neurônio único de output
- Sem ativação (regressão)
- Output: Preço normalizado previsto

### Parâmetros do Modelo

```
Total params: ~15,000
Trainable params: ~15,000

LSTM1: 50 * (4 * (50 + 1 + 1)) = 10,400
LSTM2: 50 * (4 * (50 + 50 + 1)) = 20,200
Dense: 51
```

---

## TREINAMENTO

### Configuração

```python
EPOCHS = 25
BATCH_SIZE = 32

history = modelo.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
```

**Por que poucos epochs?**:
- Dados financeiros são ruidosos
- Overfit acontece rapidamente
- 20-30 épocas geralmente suficientes

### Curva de Aprendizado

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Curva de Erro (MSE)')
plt.xlabel('Época')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
```

### Salvamento

```python
import joblib

# Salvar modelo
modelo.save('models/petr4.sa_model.keras')

# Salvar scaler (CRÍTICO!)
joblib.dump(scaler, 'models/petr4.sa_scaler.pkl')
```

---

## PREDIÇÃO

### Preparar Dados Novos

```python
# Baixar últimos 3 meses
ultimos_dados = yf.download(ticker, period='3mo', auto_adjust=True)

# Extrair últimos 60 dias
dados_fechamento = ultimos_dados['Close'].values.reshape(-1, 1)
ultimos_60_dias = dados_fechamento[-60:]

# Normalizar com scaler TREINADO
ultimos_60_dias_scaled = scaler.transform(ultimos_60_dias)

# Reshape para LSTM
X_test = np.array([ultimos_60_dias_scaled])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

### Fazer Previsão

```python
# Predição (normalizada)
preco_previsto_scaled = modelo.predict(X_test)

# Reverter normalização
preco_previsto = scaler.inverse_transform(preco_previsto_scaled)

print(f"Preço previsto para {ticker}: R$ {preco_previsto[0][0]:.2f}")
```

---

## DEPLOYMENT FLASK - SISTEMA INTELIGENTE

### Arquitetura da Aplicação

```
app.py
├── get_or_update_model()  ← Lógica inteligente
├── @app.route('/')        ← Interface
└── Modelos salvos localmente
```

### Função Inteligente: get_or_update_model()

**3 Cenários de Operação**:

#### Cenário 1: Modelo NÃO Existe (Treino Completo)

```python
if not os.path.exists(model_path) or model_age_days > 90:
    print("Treinando do zero...")
    
    # 1. Baixar dados históricos (2018 até hoje)
    dados = yf.download(ticker, start='2018-01-01', end=today_str, auto_adjust=True)
    
    # 2. Pré-processar
    dados_fechamento = dados['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_scaled = scaler.fit_transform(dados_fechamento)
    
    # 3. Criar sequências
    X_train, y_train = [], []
    for i in range(60, len(dados_scaled)):
        X_train.append(dados_scaled[i-60:i, 0])
        y_train.append(dados_scaled[i, 0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # 4. Criar e treinar modelo
    modelo = Sequential([
        Input(shape=(60, 1)),
        LSTM(units=50, return_sequences=True), Dropout(0.2),
        LSTM(units=50, return_sequences=False), Dropout(0.2),
        Dense(units=1)
    ])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    # 5. Salvar
    modelo.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    return modelo, scaler, None
```

#### Cenário 2: Modelo Existe e Está Atualizado (Hoje)

```python
model_mtime = datetime.date.fromtimestamp(os.path.getmtime(model_path))

if model_mtime >= datetime.date.today():
    print("Modelo já atualizado. Carregando...")
    
    modelo = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return modelo, scaler, None
```

#### Cenário 3: Modelo Existe mas Desatualizado (Fine-tuning)

```python
else:
    print("Fazendo fine-tuning...")
    
    # 1. Carregar modelo e scaler antigos
    scaler = joblib.load(scaler_path)
    modelo = load_model(model_path)
    
    # 2. Calcular data de início (com buffer)
    buffer_days = 5
    start_date = model_mtime - timedelta(days=buffer_days)
    
    # 3. Baixar APENAS dados novos
    novos_dados = yf.download(ticker, start=start_date_str, end=today_str)
    
    # 4. Preparar novos dados
    dados_fechamento = novos_dados['Close'].values.reshape(-1, 1)
    dados_scaled = scaler.transform(dados_fechamento)  # USA SCALER ANTIGO!
    
    # 5. Criar sequências
    X_new, y_new = [], []
    for i in range(60, len(dados_scaled)):
        X_new.append(dados_scaled[i-60:i, 0])
        y_new.append(dados_scaled[i, 0])
    
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))
    
    # 6. Fine-tuning com LR baixo
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    modelo.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)
    
    # 7. Salvar (atualiza timestamp!)
    modelo.save(model_path)
    
    return modelo, scaler, None
```

**Vantagens do Fine-tuning**:
- ✅ Não re-treina do zero (economiza tempo)
- ✅ Mantém conhecimento histórico
- ✅ Adapta-se a tendências recentes
- ✅ Learning rate baixo evita "esquecer" padrões antigos

---

## INTERFACE WEB - FLASK

### app.py - Rota Principal

```python
from flask import Flask, request, render_template

app = Flask(__name__)

# Carregar lista de empresas B3
df_empresas = pd.read_csv('data/Tickers_B3.csv', sep=';')
lista_empresas = df_empresas.apply(
    lambda row: {
        'ticker': row['Codigo'] + ".SA",
        'nome': row['Empresa']
    },
    axis=1
).tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    form_data = {'ticker': 'PETR4.SA'}
    
    if request.method == 'POST':
        ticker = request.form['ticker']
        
        # 1. Obter modelo (treino, atualização, ou carregamento)
        modelo, scaler, error_msg = get_or_update_model(ticker)
        
        if error_msg:
            prediction_result = error_msg
        else:
            # 2. Preparar dados
            ultimos_dados = yf.download(ticker, period='3mo', auto_adjust=True)
            dados_fechamento = ultimos_dados['Close'].values.reshape(-1, 1)
            ultimos_60_dias = dados_fechamento[-60:]
            ultimos_60_dias_scaled = scaler.transform(ultimos_60_dias)
            
            X_test = np.array([ultimos_60_dias_scaled])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # 3. Prever
            preco_previsto_scaled = modelo.predict(X_test)
            preco_previsto = scaler.inverse_transform(preco_previsto_scaled)
            
            prediction_result = f"{ticker}: R$ {preco_previsto[0][0]:.2f}"
    
    return render_template(
        'index.html',
        prediction=prediction_result,
        form_data=form_data,
        empresas=lista_empresas
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### templates/index.html

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>📈 Previsão de Ações LSTM</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>📈 Previsor de Ações B3 com LSTM</h1>
        <p>Previsão do próximo fechamento ajustado usando IA</p>
        
        <form method="POST">
            <label for="ticker">Selecione a ação:</label>
            <input 
                list="empresas" 
                name="ticker" 
                id="ticker" 
                placeholder="Digite o código (ex: PETR4.SA)"
                value="{{ form_data.ticker }}"
                required
            >
            
            <datalist id="empresas">
                {% for empresa in empresas %}
                <option value="{{ empresa.ticker }}">
                    {{ empresa.nome }}
                </option>
                {% endfor %}
            </datalist>
            
            <button type="submit">Prever Preço</button>
        </form>
        
        {% if prediction %}
        <div class="result">
            <h2>Resultado da Previsão</h2>
            <p class="price">{{ prediction }}</p>
            <small>* Previsão para o próximo dia útil</small>
        </div>
        {% endif %}
    </div>
</body>
</html>
```

### static/css/style.css

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    padding: 40px;
    max-width: 600px;
    width: 90%;
}

h1 {
    color: #333;
    margin-bottom: 10px;
    text-align: center;
}

form {
    margin: 30px 0;
}

input[type="text"], input[list] {
    width: 100%;
    padding: 15px;
    border: 2px solid #ddd;
    border-radius: 10px;
    font-size: 16px;
    margin: 10px 0;
}

button {
    width: 100%;
    padding: 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    cursor: pointer;
    transition: transform 0.2s;
}

button:hover {
    transform: translateY(-2px);
}

.result {
    background: #f0f0f0;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 30px;
}

.price {
    font-size: 32px;
    font-weight: bold;
    color: #667eea;
    margin: 15px 0;
}
```

---

## ESTRUTURA DE ARQUIVOS

```
Previsao_Acoes/
├── app.py                      # Backend Flask
├── requirements.txt            # Dependências
├── README.md
│
├── data/
│   └── Tickers_B3.csv         # Lista de empresas
│
├── models/                     # Modelos salvos
│   ├── petr4.sa_model.keras
│   ├── petr4.sa_scaler.pkl
│   ├── vale3.sa_model.keras
│   └── ...
│
├── notebook/
│   └── Previsao_Acoes.ipynb   # Versão acadêmica
│
├── static/
│   └── css/
│       └── style.css
│
└── templates/
    └── index.html
```

---

## LIMITAÇÕES E DISCLAIMER

### ⚠️ Importante

**Este projeto é EDUCACIONAL**. NÃO use para decisões financeiras reais.

### Por Que Previsão de Ações é Difícil?

1. **Mercados são semi-aleatórios** (Efficient Market Hypothesis)
2. **Eventos imprevisíveis** (notícias, política, pandemias)
3. **Horizonte curto** (1 dia) é muito volátil
4. **LSTM vê apenas preços**, não fundamentos da empresa

### O Que o Modelo REALMENTE Faz?

- ✅ Captura **tendências** de curto/médio prazo
- ✅ Identifica **padrões sazonais**
- ❌ NÃO prevê crashes ou rallies
- ❌ NÃO considera notícias ou eventos

### Performance Esperada

- MAE: 2-5% do valor real
- Direction Accuracy: ~55-60% (levemente acima do acaso 50%)

---

## MELHORIAS FUTURAS

### Modelo
- [ ] Adicionar features: Volume, RSI, MACD, Médias Móveis
- [ ] Multi-step prediction (próximos 5 dias)
- [ ] Attention mechanism
- [ ] Ensemble (LSTM + GRU + Transformer)

### Dados
- [ ] Incluir dados de sentimento (Twitter, notícias)
- [ ] Dados macroeconômicos (taxa Selic, dólar)
- [ ] Correlação com índices (Ibovespa, S&P500)

### Aplicação
- [ ] Gráfico interativo com Plotly
- [ ] Intervalo de confiança da previsão
- [ ] Histórico de predições vs real
- [ ] API REST para integração
- [ ] Deploy em cloud (Heroku, Render)

---

## FAQ TÉCNICO

**Q: Por que 60 dias de janela?**
A: Compromisso entre memória (capturar padrões) e overfitting. 30 dias = muito curto, 120 dias = overfit. 60 dias captura ~3 meses de tendência.

**Q: Por que return_sequences=True na primeira LSTM?**
A: Para que a segunda LSTM receba sequência completa (60 timesteps). Última LSTM usa False para retornar apenas valor final.

**Q: Como evitar data leakage no scaler?**
A: Fit apenas no treino, transform no teste. Em produção, sempre usar scaler salvo do treino, nunca fit novamente.

**Q: Por que MSE em vez de MAE?**
A: MSE penaliza erros grandes mais fortemente. Em finanças, erro grande (prever R$10 quando real é R$50) é mais grave que pequeno.

**Q: Fine-tuning funciona melhor que re-treino?**
A: Sim, geralmente. Mantém conhecimento histórico e adapta a tendências recentes. Re-treino do zero pode "esquecer" padrões antigos úteis.

**Q: Como interpretar previsão?**
A: Como **tendência**, não valor exato. Se prevê R$40 e real é R$38-42, modelo está ok. Diferença >10% indica problema.

**Q: Modelo funciona para qualquer ação?**
A: Funciona melhor para ações líquidas (alto volume) com padrões. Ações voláteis ou ilíquidas são imprevisíveis.

---

## TAGS DE BUSCA

`#lstm` `#series-temporais` `#time-series` `#previsao-acoes` `#stock-prediction` `#yfinance` `#flask` `#fine-tuning` `#b3` `#finanças` `#deep-learning` `#rnn` `#keras` `#tensorflow`

---

**Versão**: 1.0  
**Compatibilidade**: TensorFlow 2.x, Flask 2.x  
**Uso recomendado**: Educação sobre LSTM em séries temporais, exemplo de fine-tuning automático

**DISCLAIMER**: Este projeto é para fins educacionais. Não use para decisões de investimento reais. Consulte um profissional certificado.
