# %% [markdown]
# # Previsão de Demanda de Rede Farmacêutica

# %% [markdown]
# ## Índice da Análise

# %% [markdown]
# 1. Business Problem
#    * Starting Context
#      * Business Understanding
# 2. Starting Phase
#    * Requirements
#    * Libraries
#    * Notebook Settings
# 3. Analysis Phase
#    * Funções definidas
#    * Data Collection
#    * Data Cleaning
#      * Feature Extraction
#    * Data Analysis
#      * Descriptive Data Analysis
#      * Hypotheses Mindmap
#      * Exploratory Data Analysis
# 4. Model Phase
#    * Funções definidas
#    * Feature Engineering
#      * Data Preparation
#      * Feature Selection
#    * Model Building
#      * Train-Test Splitting
#      * Model Selection
#        * Baseline Training
#        * Cross Validation Training
#        * Models Performance
#    * Model Evaluation
#      * Model Hyperparameter Fine Tuning
#      * Metrics Interpretation
#        * Business Metrics
#        * Model Metrics
# 5. Deployment Phase
#    * Visualization and Dashboard
#      * Performance Assessment
#      * Model Performance
#        * Baseline vs Model Performance
#        * Model Performance in Business
#      * Business Performance Gain
#    * API Development
#      * Prediction Class
#      * API Handler
#      * API Tester
#    * Web App
#      * Frontend

# %% [markdown]
# # 1. Business Problem

# %% [markdown]
# ## Starting Context

# %% [markdown]
# "Uma empresa de distribuidora de medicamentos deseja melhorar suas previsões de demanda de produtos para otimizar seus estoques e reduzir custos operacionais. Atualmente, as previsões são feitas de forma manual e não têm sido eficazes, resultando tanto em excesso de estoque quanto em falta de produtos. Você foi convidado para desenvolver um modelo de previsão de demanda utilizando séries temporais."

# %% [markdown]
# ### Business Understanding

# %% [markdown]
# É possível entender o problema de negócio fazendo apenas 4 perguntas:
# - **Qual a Motivação?** O Representante da empresa requisitou uma análise de demanda para os próximos 12 meses.
# - **Qual a causa raiz do problema?** Melhor previsibilidade de demanda para controle de estoque e redução de custos logísticos; dificuldade em determinar a demanda mensal.
# - **Quem é o dono do problema?** O Representante da empresa distribuidora.
# - **Qual o formato da solução de deploy?**
#   - *Granularidade*: Previsão de demanda mensal para os próximos 12 meses.
#   - *Tipo de Análise*: Predição de séries temporais.
#   - *Potenciais Métodos*: ARIMA, SARIMA, SARIMAX com variáveis exógenas (Fourier).
#   - *Formato de deploy*: Predição acessada via API Telegram; entrega inicial de 12 meses.

# %% [markdown]
# # 2. Starting Phase

# %% [markdown]
# ## Requirements

# %% [markdown]
# ```
# pandas>=2.0.0
# numpy>=1.24.0
# matplotlib>=3.7.0
# seaborn>=0.12.0
# scipy>=1.10.0
# statsmodels>=0.13.0
# prophet>=1.1.0
# scikit-learn>=1.2.0
# pmdarima>=2.0.0
# ```

# %% [markdown]
# ## Libraries

# %%
try:
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from matplotlib import pyplot as plt
    import warnings
    from datetime import datetime, timedelta
    from IPython.display import Image
    from IPython.core.display import HTML
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy.stats import boxcox
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import pmdarima as pm
    warnings.filterwarnings('ignore')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print("Import successful")
except Exception as e:
    print(f"Error while importing libraries: {e}")

# %%
Start = datetime.now()

# %% [markdown]
# ## Notebook Settings

# %%
def notebook_settings():
    #%matplotlib inline
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [30, 15]
    plt.rcParams['font.size'] = 24
    #display(HTML('<style>.container { width:100% !important; }</style>'))
    sns.set_theme()

notebook_settings()

# %% [markdown]
# # 3. Analysis Phase

# %% [markdown]
# ### Funções definidas

# %%
# Funções de preparação de dados
def get_column_names(df):
    column_names = pd.Series(df.columns.values)
    columns = pd.DataFrame(column_names)
    columns.columns = ['Columns']
    return columns, column_names

def get_dimensions(df):
    dimensions_1 = pd.Series(df.shape[1])
    dimensions_2 = pd.Series(df.shape[0])
    dimensions = pd.concat([dimensions_1, dimensions_2], axis=1)
    dimensions.columns = ['Columns', 'Rows']
    dimensions = dimensions.T
    dimensions.columns = ['Dimensions']
    return dimensions

def get_dataset_types(df):
    df_types = pd.DataFrame(df.dtypes)
    df_types.reset_index(drop=True, inplace=True)
    df_types.columns = ["Data_Type"]
    df_types["Column"] = pd.Series(df.columns).values
    df_types.set_index("Column", inplace=True)
    return df_types

def get_missing_values(df):
    missing_values = pd.DataFrame(df.isna().sum())
    missing_values.reset_index(drop=True, inplace=True)
    missing_values.columns = ["Missing_Values"]
    missing_values["Column"] = pd.Series(df.columns).values
    missing_values.set_index("Column", inplace=True)
    missing_values.sort_values(by="Missing_Values", ascending=False, inplace=True)
    return missing_values

def get_num_statistics_metrics(df):
    ct1 = pd.DataFrame(df.apply(np.mean)).T
    ct2 = pd.DataFrame(df.apply(np.median)).T
    d1 = pd.DataFrame(df.apply(np.std)).T
    d2 = pd.DataFrame(df.apply(min)).T
    d3 = pd.DataFrame(df.apply(max)).T
    d4 = pd.DataFrame(df.apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(df.apply(lambda x: x.skew())).T
    d6 = pd.DataFrame(df.apply(lambda x: x.kurtosis())).T
    metrics = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
    metrics.columns = ['Attributes', 'Min', 'Max', 'Range', 'Mean', 'Median', 'Standart Deviation', 'Skew', 'Kurtosis']
    metrics.set_index('Attributes', inplace=True)
    return metrics

# Funções de análise de séries temporais 
def test_stationarity(timeseries, window=12, title='', figsize=(12, 8)):
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title(f'Análise de Estacionariedade: {title}')
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label=f'Média Móvel (janela={window})')
    plt.plot(rolling_std, color='green', label=f'Desvio Padrão Móvel (janela={window})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print('Resultados do Teste de Dickey-Fuller:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estatística de Teste', 'p-value', 'Defasagens Usadas', 'Número de Observações Usadas'])
    for key, value in dftest[4].items():
        dfoutput[f'Valor Crítico ({key})'] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusão: A série é estacionária (rejeitar a hipótese nula)")
    else:
        print("Conclusão: A série não é estacionária (falhar em rejeitar a hipótese nula)")
    return dfoutput

def plot_acf_pacf(series, figsize=(12, 8), lags=40):
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    plot_acf(series, ax=ax[0], lags=lags)
    ax[0].set_title('Função de Autocorrelação (ACF)')
    plot_pacf(series, ax=ax[1], lags=lags)
    ax[1].set_title('Função de Autocorrelação Parcial (PACF)')
    plt.tight_layout()
    plt.show()

def seasonal_decompose_plot(timeseries, model='additive', period=12, figsize=(14, 10)):
    decomposition = seasonal_decompose(timeseries, model=model, period=period)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Série Original')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Tendência')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Sazonalidade')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Resíduos')
    plt.tight_layout()
    plt.show()
    return decomposition

# %% [markdown]
# ## Data Collection

# %%
try:
    df_raw = pd.read_csv('./df_case_1_-_500_linhas.csv')
    print("Dataset carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar: {e}")

# Seleção de Produto A, Centro A
df = df_raw[(df_raw['PRODUTO'] == 'A') & (df_raw['CENTROS DE DISTRIBUICAO'] == 'A')].copy()
date_cols = [col for col in df.columns if col not in ['CATEGORIA', 'PRODUTO', 'FORNECEDOR', 'COMPRADOR', 'CENTROS DE DISTRIBUICAO', 'DESCRICAO', 'QTD DE CAIXAS']]
df_ts = pd.melt(df, id_vars=['PRODUTO'], value_vars=date_cols, var_name='date', value_name='sales')
df_ts['date'] = pd.to_datetime(df_ts['date'], format='%b%y', errors='coerce')
df_ts = df_ts[['date', 'sales']].sort_values('date').reset_index(drop=True)
print(df_ts.head())

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# ### Feature Extraction

# %%
df_ts['sales'] = df_ts['sales'].interpolate()
print(f"Valores faltantes após interpolação: {df_ts['sales'].isnull().sum()}")

# %% [markdown]
# ## Data Analysis

# %% [markdown]
# ### Descriptive Data Analysis

# %%
print(f"Dimensões: {get_dimensions(df_ts)}")
print(f"Tipos de dados:\n{get_dataset_types(df_ts)}")
print(f"Valores faltantes:\n{get_missing_values(df_ts)}")
print(f"Estatísticas numéricas:\n{get_num_statistics_metrics(df_ts[['sales']])}")

# %% [markdown]
# ### Hypotheses Mindmap

# %% [markdown]
# - H1: A demanda tem tendência crescente ao longo do tempo.
# - H2: Existe sazonalidade anual na demanda.
# - H3: Anomalias afetam os picos de demanda.

# %% [markdown]
# ### Exploratory Data Analysis

# %%
# Visualização inicial
plt.plot(df_ts['date'], df_ts['sales'], marker='o', label='Vendas')
plt.title('Vendas Mensais do Produto A (Centro A)')
plt.xlabel('Data')
plt.ylabel('Qtd de Caixas')
plt.legend()
plt.show()

# Estacionaridade
test_stationarity(df_ts['sales'], title='Série Original')

# Transformação (diferenciação e Box-Cox)
df_ts['sales_diff'] = df_ts['sales'].diff().dropna()
df_ts['sales_boxcox'], lam = boxcox(df_ts['sales'])
test_stationarity(df_ts['sales_diff'], title='Série Diferenciada')
test_stationarity(df_ts['sales_boxcox'], title='Série Box-Cox')

# Decomposição
seasonal_decompose_plot(df_ts['sales'], period=12)

# Autocorrelação
plot_acf_pacf(df_ts['sales_diff'].dropna())

# Padrões sazonais
df_ts['month'] = df_ts['date'].dt.month_name().str[:3]
sns.boxplot(x='month', y='sales', data=df_ts, order=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.title('Distribuição de Vendas por Mês')
plt.show()

# %% [markdown]
# # 4. Model Phase

# %% [markdown]
# ### Funções definidas

# %%
# Funções de validação e métricas 
def time_series_split(data, date_column, test_size=0.2):
    data = data.sort_values(by=date_column)
    split_point = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    print(f"Período de treinamento: {train_data[date_column].min()} até {train_data[date_column].max()}")
    print(f"Período de teste: {test_data[date_column].min()} até {test_data[date_column].max()}")
    return train_data, test_data

def crossval_time_series(cross_dataset, kfold, weeks, model_name, model_type, target_var, date_var, exog_vars=None, verbose=False):
    mae_list = []
    mape_list = []
    rmse_list = []
    wmape_list = []
    cross_dataset = cross_dataset.sort_values(by=date_var).reset_index(drop=True)
    for k in reversed(range(1, kfold+1)):
        if verbose:
            print(f'KFold Number: {k}')
        validation_start_date = cross_dataset[date_var].max() - timedelta(days=k*weeks*7)
        validation_end_date = cross_dataset[date_var].max() - timedelta(days=(k-1)*weeks*7)
        training = cross_dataset[cross_dataset[date_var] < validation_start_date].copy()
        validation = cross_dataset[(cross_dataset[date_var] >= validation_start_date) & 
                                  (cross_dataset[date_var] <= validation_end_date)].copy()
        if model_type.lower() == 'sarima':
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
            if exog_vars:
                model = SARIMAX(training[target_var], exog=training[exog_vars], order=order, seasonal_order=seasonal_order)
                results = model.fit(disp=False)
                y_pred = results.forecast(steps=len(validation), exog=validation[exog_vars])
            else:
                model = SARIMAX(training[target_var], order=order, seasonal_order=seasonal_order)
                results = model.fit(disp=False)
                y_pred = results.forecast(steps=len(validation))
        elif model_type.lower() == 'prophet':
            train_df = pd.DataFrame({'ds': training[date_var], 'y': training[target_var]})
            model = Prophet()
            model.fit(train_df)
            future = pd.DataFrame({'ds': validation[date_var]})
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values
        y_true = validation[target_var].values
        mae = mean_absolute_error(y_true, y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if sum(mask) > 0 else np.nan
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        wmape_list.append(wmape)
    return pd.DataFrame([{
        'Model Name': model_name,
        'MAE CV': f"{np.mean(mae_list):.2f} +/- {np.std(mae_list):.2f}",
        'MAPE CV': f"{np.mean(mape_list):.2f} +/- {np.std(mape_list):.2f}",
        'RMSE CV': f"{np.mean(rmse_list):.2f} +/- {np.std(rmse_list):.2f}",
        'WMAPE CV': f"{np.mean(wmape_list):.2f} +/- {np.std(wmape_list):.2f}"
    }])

def model_metrics(model_name, y_true, y_pred):
    def metric_round(metric):
        return float(np.round(metric, 4))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if sum(mask) > 0 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    return pd.DataFrame([{
        'Model Name': model_name,
        'MAE': metric_round(mae),
        'MAPE': metric_round(mape * 100),
        'RMSE': metric_round(rmse),
        'WMAPE': metric_round(wmape * 100)
    }])

# Funções de modelagem 
def fit_sarima(train_data, test_data, target_var, date_var, order=(1,1,1), seasonal_order=(1,1,1,12), exog_vars=None):
    y_train = train_data[target_var]
    if exog_vars:
        exog_train = train_data[exog_vars]
        exog_test = test_data[exog_vars]
        model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order)
    else:
        exog_test = None
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    print("Ajustando modelo SARIMA...")
    results = model.fit(disp=False)
    print("Modelo ajustado!")
    if exog_vars:
        y_pred = results.forecast(steps=len(test_data), exog=exog_test)
    else:
        y_pred = results.forecast(steps=len(test_data))
    y_true = test_data[target_var].values
    metrics = model_metrics("SARIMA", y_true, y_pred)
    plt.figure(figsize=(12, 6))
    plt.plot(train_data[date_var], train_data[target_var], label='Treino')
    plt.plot(test_data[date_var], test_data[target_var], label='Teste (Real)')
    plt.plot(test_data[date_var], y_pred, label='Previsão', color='red')
    plt.title('Previsão SARIMA')
    plt.legend()
    plt.show()
    return results, y_pred, metrics

def fit_prophet(train_data, test_data, target_var, date_var, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
    df_train = pd.DataFrame({'ds': train_data[date_var], 'y': train_data[target_var]})
    model = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
    model.fit(df_train)
    future = pd.DataFrame({'ds': test_data[date_var]})
    forecast = model.predict(future)
    y_true = test_data[target_var].values
    y_pred = forecast['yhat'].values
    metrics = model_metrics("Prophet", y_true, y_pred)
    plt.figure(figsize=(12, 6))
    plt.plot(train_data[date_var], train_data[target_var], label='Treino')
    plt.plot(test_data[date_var], test_data[target_var], label='Teste (Real)')
    plt.plot(test_data[date_var], y_pred, label='Previsão', color='red')
    plt.title('Previsão Prophet')
    plt.legend()
    plt.show()
    return model, forecast, metrics

def find_optimal_sarima_params(train_data, target_var, p_range=(0, 2), d_range=(0, 2), q_range=(0, 2), P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2), s=12, exog_vars=None):
    y_train = train_data[target_var]
    if exog_vars:
        exog_train = train_data[exog_vars]
    else:
        exog_train = None
    print("Iniciando busca automática de parâmetros SARIMA...")
    auto_arima = pm.auto_arima(y_train, exogenous=exog_train, start_p=p_range[0], max_p=p_range[1], start_d=d_range[0], max_d=d_range[1], start_q=q_range[0], max_q=q_range[1], start_P=P_range[0], max_P=P_range[1], start_D=D_range[0], max_D=D_range[1], start_Q=Q_range[0], max_Q=Q_range[1], m=s, seasonal=True, error_action='ignore', suppress_warnings=True, stepwise=True, trace=True)
    print(f"Melhor ordem SARIMA: {auto_arima.order} {auto_arima.seasonal_order}")
    return auto_arima.order, auto_arima.seasonal_order

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Data Preparation

# %%
# Fourier terms como variáveis exógenas
for order in range(1, 4):
    df_ts[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * df_ts['date'].dt.month / 12)
    df_ts[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * df_ts['date'].dt.month / 12)
exog_vars = [col for col in df_ts.columns if 'fourier' in col]

# %% [markdown]
# ### Feature Selection

# %%
# Usaremos todas as variáveis Fourier para SARIMAX
print(f"Variáveis exógenas selecionadas: {exog_vars}")

# %% [markdown]
# ## Model Building

# %% [markdown]
# ### Train-Test Splitting

# %%
train, test = time_series_split(df_ts, 'date', test_size=0.2)

# %% [markdown]
# ### Model Selection

# %% [markdown]
# #### Baseline Training

# %%
baseline_pred = train['sales'].mean()
baseline_metrics = model_metrics("Baseline", test['sales'], [baseline_pred] * len(test))
print(baseline_metrics)

# %% [markdown]
# #### Cross Validation Training

# %%
sarima_cv = crossval_time_series(df_ts, kfold=3, weeks=4, model_name='SARIMA', model_type='sarima', target_var='sales', date_var='date', exog_vars=exog_vars)
prophet_cv = crossval_time_series(df_ts, kfold=3, weeks=4, model_name='Prophet', model_type='prophet', target_var='sales', date_var='date')
print(sarima_cv)
print(prophet_cv)

# %% [markdown]
# #### Models Performance

# %%
sarima_results, sarima_pred, sarima_metrics = fit_sarima(train, test, 'sales', 'date', exog_vars=exog_vars)
prophet_model, prophet_forecast, prophet_metrics = fit_prophet(train, test, 'sales', 'date')

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# ### Model Hyperparameter Fine Tuning

# %%
order, seasonal_order = find_optimal_sarima_params(train, 'sales', exog_vars=exog_vars)

# %% [markdown]
# ### Metrics Interpretation

# %% [markdown]
# #### Business Metrics

# %% [markdown]
# "WMAPE é a métrica principal, pois pondera erros pelo volume, essencial para estoque. RMSE destaca outliers relevantes."

# %% [markdown]
# #### Model Metrics

# %%
print(pd.concat([baseline_metrics, sarima_metrics, prophet_metrics]))

# %% [markdown]
# # 5. Deployment Phase

# %% [markdown]
# ## Visualization and Dashboard

# %% [markdown]
# ### Performance Assessment

# %% [markdown]
# ### Model Performance

# %% [markdown]
# #### Baseline vs Model Performance

# %%
# <graph comparing baseline, SARIMA, Prophet here>

# %% [markdown]
# #### Model Performance in Business

# %%
# <graph showing business impact here>

# %% [markdown]
# ### Business Performance Gain

# %% [markdown]
# "Redução de custos estimada em X% com estoque otimizado."

# %% [markdown]
# ## API Development

# %% [markdown]
# ### Prediction Class

# %% [markdown]
# ### API Handler

# %% [markdown]
# ### API Tester

# %% [markdown]
# ## Web App

# %% [markdown]
# ### Frontend

# %% [markdown]
# "API Telegram para acesso via celular planejada como próxima etapa."