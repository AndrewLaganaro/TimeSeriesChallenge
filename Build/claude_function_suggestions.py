import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import datetime
from datetime import timedelta
import warnings
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

# ------------------ FUNÇÕES DE PREPARAÇÃO DE DADOS ------------------

def get_column_names(df):
    """
    Retorna os nomes das colunas do DataFrame.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        DataFrame e Series com os nomes das colunas
    """
    column_names = pd.Series(df.columns.values)
    columns = pd.DataFrame(column_names)
    columns.columns = ['Columns']
    return columns, column_names


def get_dimensions(df):
    """
    Retorna as dimensões do DataFrame.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        DataFrame com o número de linhas e colunas
    """
    dimensions_1 = pd.Series(df.shape[1])
    dimensions_2 = pd.Series(df.shape[0])
    dimensions = pd.concat([dimensions_1, dimensions_2], axis=1)
    dimensions.columns = ['Columns', 'Rows']
    dimensions = dimensions.T
    dimensions.columns = ['Dimensions'] 
    return dimensions


def get_dataset_types(df):
    """
    Retorna os tipos de dados de cada coluna do DataFrame.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        DataFrame com os tipos de dados de cada coluna
    """
    df_types = pd.DataFrame(df.dtypes)
    df_types.reset_index(drop=True, inplace=True)
    df_types.columns = ["Data_Type"]
    df_types["Column"] = pd.Series(df.columns).values
    df_types.set_index("Column", inplace=True)
    return df_types


def get_missing_values(df):
    """
    Retorna o número de valores faltantes em cada coluna do DataFrame.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        DataFrame com o número de valores faltantes em cada coluna
    """
    missing_values = pd.DataFrame(df.isna().sum())
    missing_values.reset_index(drop=True, inplace=True)
    missing_values.columns = ["Missing_Values"]
    missing_values["Column"] = pd.Series(df.columns).values
    missing_values.set_index("Column", inplace=True)
    missing_values.sort_values(by="Missing_Values", ascending=False, inplace=True)
    
    return missing_values


def get_broadview_miss_val(df):
    """
    Fornece uma visão detalhada dos valores faltantes no DataFrame.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        DataFrame com detalhes dos valores faltantes e uma lista com nomes das colunas que têm valores faltantes
    """
    missing_values = pd.DataFrame(df.isna().sum()/df.shape[0])
    missing_values.reset_index(drop=True, inplace=True)
    
    missing_values.columns = ["Absolute Missing (%)"]
    missing_values["Column"] = pd.Series(df.columns).values
    missing_values.reset_index(drop=True, inplace=True)
    missing_values.set_index("Column", inplace=True)
    
    # Calculating the percentage of missing values in relation to the total number of rows in the column
    column_missing = []
    column_total = []
    column_miss = []
    for column in df.columns:
        col_miss = df[column].isnull().sum()
        total_values = df[column].count()
        
        column_miss.append(col_miss)
        column_total.append(total_values) 
        column_missing.append((col_miss/total_values))
    
    missing_values["Column Missing (%)"] = pd.Series(column_missing).values
    missing_values["Column Remaining (%)"] = -(pd.Series(column_missing).values-1)
    missing_values["Column Total"] = pd.Series(column_total).values
    missing_values["Column Missing"] = pd.Series(column_miss).values
    
    missing_values.sort_values(by="Absolute Missing (%)", ascending=False, inplace=True)
    missing_columns = list(missing_values.query("`Absolute Missing (%)` > 0").index)

    return missing_values, missing_columns


def get_num_statistics_metrics(df):
    """
    Calcula métricas estatísticas para colunas numéricas.
    
    Args:
        df: DataFrame pandas com colunas numéricas
        
    Returns:
        DataFrame com métricas estatísticas
    """
    # Central Tendency - mean, median 
    ct1 = pd.DataFrame(df.apply(np.mean)).T
    ct2 = pd.DataFrame(df.apply(np.median)).T
    # Dispersion Metrics - std, min, max, range, skew, kurtosis
    d1 = pd.DataFrame(df.apply(np.std)).T
    d2 = pd.DataFrame(df.apply(min)).T
    d3 = pd.DataFrame(df.apply(max)).T
    d4 = pd.DataFrame(df.apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(df.apply(lambda x: x.skew())).T
    d6 = pd.DataFrame(df.apply(lambda x: x.kurtosis())).T

    # concatenar
    metrics = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
    metrics.columns = ['Attributes', 'Min', 'Max', 'Range', 'Mean', 'Median', 'Standart Deviation', 'Skew', 'Kurtosis']
    metrics.set_index('Attributes', inplace=True)
    return metrics


# ------------------ FUNÇÕES DE VALIDAÇÃO CRUZADA E MÉTRICAS ------------------

def time_series_split(data, date_column, test_size=0.2):
    """
    Divide os dados em treino e teste respeitando a ordem temporal.
    
    Args:
        data: DataFrame pandas
        date_column: Nome da coluna de data
        test_size: Proporção dos dados para teste
        
    Returns:
        train_data, test_data: DataFrames de treino e teste
    """
    data = data.sort_values(by=date_column)
    split_point = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    
    print(f"Período de treinamento: {train_data[date_column].min()} até {train_data[date_column].max()}")
    print(f"Período de teste: {test_data[date_column].min()} até {test_data[date_column].max()}")
    
    return train_data, test_data


def crossval_time_series(cross_dataset, kfold, weeks, model_name, model_type, target_var, date_var, exog_vars=None, verbose=False):
    """
    Realiza validação cruzada para séries temporais.
    
    Args:
        cross_dataset: DataFrame com os dados
        kfold: Número de folds
        weeks: Número de semanas em cada fold
        model_name: Nome do modelo para identificação
        model_type: Tipo do modelo ('sarima', 'arima', 'prophet')
        target_var: Nome da variável alvo
        date_var: Nome da variável de data
        exog_vars: Lista de variáveis exógenas (para SARIMA/ARIMA)
        verbose: Se True, imprime informações durante o processo
        
    Returns:
        DataFrame com resultados da validação cruzada
    """
    mae_list = []
    mape_list = []
    rmse_list = []
    wmape_list = []
    
    # Ordenar o dataset pela data
    cross_dataset = cross_dataset.sort_values(by=date_var).reset_index(drop=True)
    
    for k in reversed(range(1, kfold+1)):
        if verbose:
            print(f'KFold Number: {k}')
        
        # Definir datas de início e fim para validação
        validation_start_date = cross_dataset[date_var].max() - timedelta(days=k*weeks*7)
        validation_end_date = cross_dataset[date_var].max() - timedelta(days=(k-1)*weeks*7)
        
        # Filtrar datasets
        training = cross_dataset[cross_dataset[date_var] < validation_start_date].copy()
        validation = cross_dataset[(cross_dataset[date_var] >= validation_start_date) & 
                                  (cross_dataset[date_var] <= validation_end_date)].copy()
        
        # Treinar o modelo
        if model_type.lower() == 'sarima':
            # Parâmetros SARIMA (normalmente seriam otimizados)
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)  # 12 para sazonalidade mensal
            
            if exog_vars is not None:
                exog_train = training[exog_vars]
                exog_val = validation[exog_vars]
                model = SARIMAX(training[target_var], 
                               exog=exog_train, 
                               order=order, 
                               seasonal_order=seasonal_order)
                results = model.fit(disp=False)
                y_pred = results.forecast(steps=len(validation), exog=exog_val)
            else:
                model = SARIMAX(training[target_var], 
                               order=order, 
                               seasonal_order=seasonal_order)
                results = model.fit(disp=False)
                y_pred = results.forecast(steps=len(validation))
                
        elif model_type.lower() == 'arima':
            # Parâmetros ARIMA (normalmente seriam otimizados)
            order = (1, 1, 1)
            
            if exog_vars is not None:
                exog_train = training[exog_vars]
                exog_val = validation[exog_vars]
                model = SARIMAX(training[target_var], 
                               exog=exog_train, 
                               order=order)
                results = model.fit(disp=False)
                y_pred = results.forecast(steps=len(validation), exog=exog_val)
            else:
                model = SARIMAX(training[target_var], 
                               order=order)
                results = model.fit(disp=False)
                y_pred = results.forecast(steps=len(validation))
                
        elif model_type.lower() == 'prophet':
            # Preparar dados para Prophet
            train_df = pd.DataFrame({
                'ds': training[date_var],
                'y': training[target_var]
            })
            
            val_dates = validation[date_var]
            
            # Treinar modelo Prophet
            model = Prophet()
            model.fit(train_df)
            
            # Fazer previsão
            future = pd.DataFrame({'ds': val_dates})
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values
            
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        
        # Avaliar modelo
        y_true = validation[target_var].values
        
        # Calcular métricas
        mae = mean_absolute_error(y_true, y_pred)
        
        # Evitar divisão por zero no MAPE
        mask = y_true != 0
        if sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
            
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # WMAPE - Weighted Mean Absolute Percentage Error
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        
        # Armazenar resultados
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        wmape_list.append(wmape)
        
    # Formatar resultados
    cross_val_results = {
        'Model Name': model_name,
        'MAE CV': f"{np.mean(mae_list):.2f} +/- {np.std(mae_list):.2f}",
        'MAPE CV': f"{np.mean(mape_list):.2f} +/- {np.std(mape_list):.2f}",
        'RMSE CV': f"{np.mean(rmse_list):.2f} +/- {np.std(rmse_list):.2f}",
        'WMAPE CV': f"{np.mean(wmape_list):.2f} +/- {np.std(wmape_list):.2f}"
    }
    
    cross_val = pd.DataFrame([cross_val_results])
    
    return cross_val


def model_metrics(model_name, y_true, y_pred):
    """
    Calcula métricas de avaliação para modelos de séries temporais.
    
    Args:
        model_name: Nome do modelo
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        DataFrame com métricas de avaliação
    """
    # Função para arredondar métricas
    def metric_round(metric):
        return float(np.round(metric, 4))
    
    # Mean Percentage Error
    def mean_percentage_error(y_true, y_pred):
        # Remover linhas com vendas zero para evitar divisão por zero
        mask = y_true != 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        if len(y_true_filtered) > 0:
            mpe = np.mean((y_true_filtered - y_pred_filtered) / y_true_filtered)
            return mpe
        else:
            return np.nan
    
    # Calcular métricas
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE - evitar divisão por zero
    mask = y_true != 0
    if sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        mape = np.nan
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mpe = mean_percentage_error(y_true, y_pred)
    
    # WMAPE - Weighted Mean Absolute Percentage Error
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    
    # Organizar resultados
    model_metrics = {
        'Model Name': model_name, 
        'MAE': metric_round(mae), 
        'MAPE': metric_round(mape * 100),  # Converter para percentual
        'RMSE': metric_round(rmse),
        'MPE': metric_round(mpe * 100),    # Converter para percentual
        'WMAPE': metric_round(wmape * 100) # Converter para percentual
    }
    
    metrics = pd.DataFrame([model_metrics])
    
    return metrics


# ------------------ FUNÇÕES DE ANÁLISE DE SÉRIES TEMPORAIS ------------------

def test_stationarity(timeseries, window=12, title='', figsize=(12, 8)):
    """
    Teste de estacionariedade para séries temporais com o teste de Dickey-Fuller.
    
    Args:
        timeseries: Série temporal a ser testada
        window: Janela para calcular estatísticas móveis
        title: Título do gráfico
        figsize: Tamanho da figura
        
    Returns:
        Resultado do teste de Dickey-Fuller
    """
    # Calcular estatísticas móveis
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    # Plotar estatísticas móveis
    plt.figure(figsize=figsize)
    plt.title(f'Análise de Estacionariedade: {title}')
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label=f'Média Móvel (janela={window})')
    plt.plot(rolling_std, color='green', label=f'Desvio Padrão Móvel (janela={window})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Teste de Dickey-Fuller
    print('Resultados do Teste de Dickey-Fuller:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    
    dfoutput = pd.Series(
        dftest[0:4], 
        index=['Estatística de Teste', 'p-value', 'Defasagens Usadas', 'Número de Observações Usadas']
    )
    
    for key, value in dftest[4].items():
        dfoutput[f'Valor Crítico ({key})'] = value
    
    print(dfoutput)
    
    # Interpretar resultado
    if dftest[1] <= 0.05:
        print("Conclusão: A série é estacionária (rejeitar a hipótese nula)")
    else:
        print("Conclusão: A série não é estacionária (falhar em rejeitar a hipótese nula)")
        
    return dfoutput


def plot_acf_pacf(series, figsize=(12, 8), lags=40):
    """
    Plota as funções de autocorrelação (ACF) e autocorrelação parcial (PACF).
    
    Args:
        series: Série temporal
        figsize: Tamanho da figura
        lags: Número de defasagens a serem plotadas
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    
    # ACF
    plot_acf(series, ax=ax[0], lags=lags)
    ax[0].set_title('Função de Autocorrelação (ACF)')
    
    # PACF
    plot_pacf(series, ax=ax[1], lags=lags)
    ax[1].set_title('Função de Autocorrelação Parcial (PACF)')
    
    plt.tight_layout()
    plt.show()


def seasonal_decompose_plot(timeseries, model='multiplicative', period=None, figsize=(14, 10)):
    """
    Decompõe uma série temporal em seus componentes de tendência, sazonalidade e resíduo.
    
    Args:
        timeseries: Série temporal a ser decomposta
        model: Modelo de decomposição ('multiplicative' ou 'additive')
        period: Período de sazonalidade (ex: 12 para dados mensais)
        figsize: Tamanho da figura
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if period is None:
        # Tentar detectar automaticamente (12 para mensal, 4 para trimestral, etc.)
        freq = pd.infer_freq(timeseries.index)
        if freq:
            if 'M' in freq:  # Mensal
                period = 12
            elif 'Q' in freq:  # Trimestral
                period = 4
            elif 'D' in freq:  # Diário
                period = 7
            elif 'B' in freq:  # Dias úteis
                period = 5
            else:
                period = 12  # Padrão
        else:
            period = 12  # Padrão
    
    # Decomposição
    decomposition = seasonal_decompose(timeseries, model=model, period=period)
    
    # Plotar resultados
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


def analyze_residuals(residuals, figsize=(14, 10)):
    """
    Analisa os resíduos de um modelo para verificar pressupostos.
    
    Args:
        residuals: Série de resíduos
        figsize: Tamanho da figura
    """
    from scipy import stats
    import statsmodels.stats.diagnostic as smd
    from statsmodels.stats.stattools import durbin_watson
    
    # Plotar resíduos
    plt.figure(figsize=figsize)
    
    # Subplot 1: Resíduos ao longo do tempo
    plt.subplot(2, 2, 1)
    plt.plot(residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Resíduos ao Longo do Tempo')
    
    # Subplot 2: Histograma dos resíduos
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')
    
    # Adicionar curva de distribuição normal
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Histograma dos Resíduos vs. Normal')
    
    # Subplot 3: QQ Plot
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Subplot 4: Autocorrelação dos resíduos
    plt.subplot(2, 2, 4)
    plot_acf(residuals.dropna(), lags=40)
    plt.title('Autocorrelação dos Resíduos')
    
    plt.tight_layout()
    plt.show()
    
    # Teste de normalidade (Shapiro-Wilk)
    print("Teste de Normalidade dos Resíduos (Shapiro-Wilk):")
    stat, p_value = stats.shapiro(residuals.dropna())
    print(f"Estatística de teste: {stat:.4f}")
    print(f"P-valor: {p_value:.4f}")
    if p_value > 0.05:
        print("Os resíduos parecem seguir uma distribuição normal (falha em rejeitar H0)")
    else:
        print("Os resíduos não seguem uma distribuição normal (rejeitar H0)")
    
    # Teste de independência (Durbin-Watson)
    print("\nTeste de Independência dos Resíduos (Durbin-Watson):")
    dw_stat = durbin_watson(residuals.dropna())
    print(f"Estatística Durbin-Watson: {dw_stat:.4f}")
    if dw_stat < 1.5:
        print("Possível autocorrelação positiva")
    elif dw_stat > 2.5:
        print("Possível autocorrelação negativa")
    else:
        print("Não há evidência de autocorrelação (próximo de 2)")
    
    # Teste de heterocedasticidade (Breusch-Pagan)
    print("\nTeste de Heterocedasticidade (Breusch-Pagan):")
    try:
        # Preparar dados
        residuals_squared = residuals.dropna() ** 2
        x = np.ones(len(residuals_squared))
        
        bp_test = smd.het_breuschpagan(residuals_squared, x.reshape(-1, 1))
        
        print(f"Estatística de teste: {bp_test[0]:.4f}")
        print(f"P-valor: {bp_test[1]:.4f}")
        
        if bp_test[1] > 0.05:
            print("Não há evidência de heterocedasticidade (falha em rejeitar H0)")
        else:
            print("Há evidência de heterocedasticidade (rejeitar H0)")
    except:
        print("Não foi possível realizar o teste Breusch-Pagan com os dados fornecidos")


# ------------------ FUNÇÕES PARA MODELOS ESPECÍFICOS ------------------

def fit_sarima(train_data, test_data, target_var, date_var, order=(1,1,1), seasonal_order=(1,1,1,12), exog_vars=None):
    """
    Ajusta um modelo SARIMA e faz previsões.
    
    Args:
        train_data: DataFrame de treinamento
        test_data: DataFrame de teste
        target_var: Nome da variável alvo
        date_var: Nome da variável de data
        order: Ordem do modelo ARIMA (p,d,q)
        seasonal_order: Ordem sazonal (P,D,Q,s)
        exog_vars: Lista de variáveis exógenas
        
    Returns:
        Resultados do modelo, previsões e métricas
    """
    # Preparar dados de treinamento
    y_train = train_data[target_var]
    
    # Preparar dados exógenos se fornecidos
    if exog_vars is not None:
        exog_train = train_data[exog_vars]
        exog_test = test_data[exog_vars]
        model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order)
    else:
        exog_test = None
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    
    # Ajustar o modelo
    print("Ajustando modelo SARIMA...")
    results = model.fit(disp=False)
    print("Modelo ajustado!")
    
    # Resumo do modelo
    print("\nResumo do modelo:")
    print(results.summary())
    
    # Fazer previsões
    print("\nRealizando previsões...")
    if exog_vars is not None:
        y_pred = results.forecast(steps=len(test_data), exog=exog_test)
    else:
        y_pred = results.forecast(steps=len(test_data))
    
    # Calcular métricas
    y_true = test_data[target_var].values
    metrics = model_metrics("SARIMA", y_true, y_pred)
    
    # Plotar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(train_data[date_var], train_data[target_var], label='Treino')
    plt.plot(test_data[date_var], test_data[target_var], label='Teste (Real)')
    plt.plot(test_data[date_var], y_pred, label='Previsão', color='red')
    plt.title('Previsão SARIMA')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Análise de resíduos
    residuals = pd.Series(results.resid, index=train_data.index)
    print("\nAnálise de resíduos:")
    analyze_residuals(residuals)
    
    return results, y_pred, metrics


def fit_arima(train_data, test_data, target_var, date_var, order=(1,1,1), exog_vars=None):
    """
    Ajusta um modelo ARIMA e faz previsões.
    
    Args:
        train_data: DataFrame de treinamento
        test_data: DataFrame de teste
        target_var: Nome da variável alvo
        date_var: Nome da variável de data
        order: Ordem do modelo ARIMA (p,d,q)
        exog_vars: Lista de variáveis exógenas
        
    Returns:
        Resultados do modelo, previsões e métricas
    """
    # Preparar dados de treinamento
    y_train = train_data[target_var]
    
    # Preparar dados exógenos se fornecidos
    if exog_vars is not None:
        exog_train = train_data[exog_vars]
        exog_test = test_data[exog_vars]
        model = SARIMAX(y_train, exog=exog_train, order=order)
    else:
        exog_test = None
        model = SARIMAX(y_train, order=order)
    
    # Ajustar o modelo
    print("Ajustando modelo ARIMA...")
    results = model.fit(disp=False)
    print("Modelo ajustado!")
    
    # Resumo do modelo
    print("\nResumo do modelo:")
    print(results.summary())
    
    # Fazer previsões
    print("\nRealizando previsões...")
    if exog_vars is not None:
        y_pred = results.forecast(steps=len(test_data), exog=exog_test)
    else:
        y_pred = results.forecast(steps=len(test_data))
    
    # Calcular métricas
    y_true = test_data[target_var].values
    metrics = model_metrics("ARIMA", y_true, y_pred)
    
    # Plotar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(train_data[date_var], train_data[target_var], label='Treino')
    plt.plot(test_data[date_var], test_data[target_var], label='Teste (Real)')
    plt.plot(test_data[date_var], y_pred, label='Previsão', color='red')
    plt.title('Previsão ARIMA')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Análise de resíduos
    residuals = pd.Series(results.resid, index=train_data.index)
    print("\nAnálise de resíduos:")
    analyze_residuals(residuals)
    
    return results, y_pred, metrics


def fit_prophet(train_data, test_data, target_var, date_var, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05):
    """
    Ajusta um modelo Prophet e faz previsões.
    
    Args:
        train_data: DataFrame de treinamento
        test_data: DataFrame de teste
        target_var: Nome da variável alvo
        date_var: Nome da variável de data
        yearly_seasonality: Considerar sazonalidade anual
        weekly_seasonality: Considerar sazonalidade semanal
        daily_seasonality: Considerar sazonalidade diária
        changepoint_prior_scale: Controla a flexibilidade da tendência
        
    Returns:
        Modelo, previsões e métricas
    """
    # Preparar dados no formato do Prophet
    df_train = pd.DataFrame({
        'ds': train_data[date_var],
        'y': train_data[target_var]
    })
    
    # Ajustar o modelo
    print("Ajustando modelo Prophet...")
    model = Prophet(yearly_seasonality=yearly_seasonality,
                   weekly_seasonality=weekly_seasonality,
                   daily_seasonality=daily_seasonality,
                   changepoint_prior_scale=changepoint_prior_scale)
    
    model.fit(df_train)
    print("Modelo ajustado!")
    
    # Preparar datas para previsão
    future = pd.DataFrame({'ds': test_data[date_var]})
    
    # Fazer previsões
    print("\nRealizando previsões...")
    forecast = model.predict(future)
    
    # Calcular métricas
    y_true = test_data[target_var].values
    y_pred = forecast['yhat'].values
    metrics = model_metrics("Prophet", y_true, y_pred)
    
    # Plotar resultados
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(train_data[date_var], train_data[target_var], label='Treino')
    plt.plot(test_data[date_var], test_data[target_var], label='Teste (Real)')
    plt.plot(test_data[date_var], y_pred, label='Previsão', color='red')
    plt.fill_between(test_data[date_var], 
                     forecast['yhat_lower'].values, 
                     forecast['yhat_upper'].values, 
                     color='red', alpha=0.2, label='Intervalo de Confiança 80%')
    plt.title('Previsão Prophet')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Componentes do modelo
    print("\nComponentes do modelo:")
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.show()
    
    # Análise de resíduos
    residuals = pd.Series(y_true - y_pred, index=test_data.index)
    print("\nAnálise de resíduos:")
    analyze_residuals(residuals)
    
    return model, forecast, metrics


def find_optimal_sarima_params(train_data, target_var, p_range=(0, 2), d_range=(0, 2), q_range=(0, 2), 
                              P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2), s=12, exog_vars=None):
    """
    Encontra os parâmetros ótimos para o modelo SARIMA.
    
    Args:
        train_data: DataFrame de treinamento
        target_var: Nome da variável alvo
        p_range, d_range, q_range: Ranges para os parâmetros do componente não-sazonal
        P_range, D_range, Q_range: Ranges para os parâmetros do componente sazonal
        s: Período sazonal
        exog_vars: Lista de variáveis exógenas
        
    Returns:
        Ordem ótima (p,d,q) e ordem sazonal ótima (P,D,Q,s)
    """
    try:
        import pmdarima as pm
        
        # Preparar dados de treinamento
        y_train = train_data[target_var]
        
        # Preparar dados exógenos se fornecidos
        if exog_vars is not None:
            exog_train = train_data[exog_vars]
        else:
            exog_train = None
        
        # Realizar busca automática de parâmetros
        print("Iniciando busca automática de parâmetros SARIMA (pode demorar um pouco)...")
        
        auto_arima = pm.auto_arima(y_train, exogenous=exog_train,
                                 start_p=p_range[0], max_p=p_range[1],
                                 start_d=d_range[0], max_d=d_range[1],
                                 start_q=q_range[0], max_q=q_range[1],
                                 start_P=P_range[0], max_P=P_range[1],
                                 start_D=D_range[0], max_D=D_range[1],
                                 start_Q=Q_range[0], max_Q=Q_range[1],
                                 m=s,
                                 seasonal=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True,
                                 trace=True)
        
        print("Busca concluída!")
        print(f"Melhor ordem SARIMA: {auto_arima.order} {auto_arima.seasonal_order}")
        
        # Resumo do modelo
        print("\nResumo do modelo:")
        print(auto_arima.summary())
        
        return auto_arima.order, auto_arima.seasonal_order
        
    except ImportError:
        print("A biblioteca pmdarima não está instalada. Instale usando 'pip install pmdarima'.")
        return None, None


def compare_models(models_metrics, figsize=(10, 6)):
    """
    Compara os resultados de diferentes modelos.
    
    Args:
        models_metrics: Lista de DataFrames com métricas dos modelos
        figsize: Tamanho da figura
    """
    # Combinar métricas
    combined_metrics = pd.concat(models_metrics).reset_index(drop=True)
    
    # Definir métricas para comparação
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'WMAPE']
    
    # Plotar comparação
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='Model Name', y=metric, data=combined_metrics, ax=axes[i])
        axes[i].set_title(f'Comparação de {metric}')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Tabela de comparação
    print("Tabela de Comparação de Modelos:")
    comparison_table = combined_metrics.set_index('Model Name')
    print(comparison_table)
    
    # Identificar melhor modelo
    best_model_wmape = combined_metrics.loc[combined_metrics['WMAPE'].idxmin()]['Model Name']
    best_model_rmse = combined_metrics.loc[combined_metrics['RMSE'].idxmin()]['Model Name']
    
    print(f"\nMelhor modelo segundo WMAPE: {best_model_wmape}")
    print(f"Melhor modelo segundo RMSE: {best_model_rmse}")
    
    return comparison_table


# ------------------ FUNÇÕES PARA ANÁLISE HIERÁRQUICA ------------------

def create_hierarchical_data(data, date_column, target_column, hierarchy_columns):
    """
    Cria estrutura de dados para modelagem hierárquica.
    
    Args:
        data: DataFrame com os dados
        date_column: Nome da coluna de data
        target_column: Nome da coluna alvo
        hierarchy_columns: Lista de colunas que definem a hierarquia (do mais alto para o mais baixo)
        
    Returns:
        DataFrame estruturado hierarquicamente
    """
    # Ordenar dados por data
    data = data.sort_values(by=date_column)
    
    # Criar nível mais alto (Total)
    total = data.groupby(date_column)[target_column].sum().reset_index()
    total['level'] = 'Total'
    
    # Criar níveis intermediários
    hierarchical_data = [total]
    
    for i, col in enumerate(hierarchy_columns):
        # Agrupar pelo nível atual
        level_data = data.groupby([date_column, col])[target_column].sum().reset_index()
        level_data['level'] = col
        level_data['group'] = level_data[col].astype(str)
        hierarchical_data.append(level_data)
        
        # Se não for o último nível, criar níveis combinados
        if i < len(hierarchy_columns) - 1:
            cols_to_group = hierarchy_columns[:i+1]
            level_combined = data.groupby([date_column] + cols_to_group)[target_column].sum().reset_index()
            level_combined['level'] = '_'.join(cols_to_group)
            level_combined['group'] = level_combined[cols_to_group].astype(str).agg('_'.join, axis=1)
            hierarchical_data.append(level_combined)
    
    return hierarchical_data


def top_down_forecast(hierarchical_data, date_column, target_column, forecast_periods=12, train_size=0.8):
    """
    Realiza previsão hierárquica top-down.
    
    Args:
        hierarchical_data: Lista de DataFrames com estrutura hierárquica
        date_column: Nome da coluna de data
        target_column: Nome da coluna alvo
        forecast_periods: Número de períodos a prever
        train_size: Proporção dos dados para treino
        
    Returns:
        Previsões em todos os níveis da hierarquia
    """
    from prophet import Prophet
    
    # Dividir dados em treino e teste
    forecasts = []
    
    # Prever o total (nível mais alto)
    total_data = hierarchical_data[0]
    
    # Determinar ponto de divisão para treino/teste
    split_point = int(len(total_data) * train_size)
    train_total = total_data.iloc[:split_point].copy()
    test_total = total_data.iloc[split_point:].copy()
    
    # Preparar dados para Prophet
    prophet_df = pd.DataFrame({
        'ds': train_total[date_column],
        'y': train_total[target_column]
    })
    
    # Treinar modelo
    model = Prophet()
    model.fit(prophet_df)
    
    # Preparar futuro
    future = pd.DataFrame({'ds': test_total[date_column]})
    
    # Prever
    forecast = model.predict(future)
    
    # Armazenar previsão do total
    forecasts.append({
        'level': 'Total',
        'group': 'Total',
        'forecast': forecast['yhat'].values,
        'actual': test_total[target_column].values
    })
    
    # Calcular proporções para os níveis inferiores
    for i in range(1, len(hierarchical_data)):
        level_data = hierarchical_data[i]
        level_name = level_data['level'].iloc[0]
        
        # Obter grupos únicos neste nível
        groups = level_data['group'].unique()
        
        # Para cada grupo, calcular sua proporção do total
        for group in groups:
            group_data = level_data[level_data['group'] == group]
            
            # Calcular proporção média no conjunto de treinamento
            group_train = group_data.iloc[:split_point]
            total_train = train_total.copy()
            
            # Alinhar datas
            merged = pd.merge(group_train, total_train, on=date_column)
            proportion = merged[f'{target_column}_x'].sum() / merged[f'{target_column}_y'].sum()
            
            # Usar proporção para prever
            group_forecast = forecasts[0]['forecast'] * proportion
            
            # Obter dados reais para teste
            group_test = group_data.iloc[split_point:]
            
            # Armazenar previsão
            forecasts.append({
                'level': level_name,
                'group': group,
                'forecast': group_forecast,
                'actual': group_test[target_column].values
            })
    
    return forecasts


def evaluate_hierarchical_forecast(forecasts):
    """
    Avalia previsões hierárquicas.
    
    Args:
        forecasts: Lista de dicionários com previsões hierárquicas
        
    Returns:
        DataFrame com métricas para cada nível e grupo
    """
    results = []
    
    for forecast in forecasts:
        y_true = forecast['actual']
        y_pred = forecast['forecast']
        
        # Calcular métricas
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        
        results.append({
            'Level': forecast['level'],
            'Group': forecast['group'],
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'WMAPE': wmape
        })
    
    return pd.DataFrame(results)


def plot_hierarchical_forecast(forecasts, date_column, test_dates):
    """
    Plota previsões hierárquicas.
    
    Args:
        forecasts: Lista de dicionários com previsões hierárquicas
        date_column: Nome da coluna de data
        test_dates: Datas para o período de teste
    """
    total_levels = len(set([f['level'] for f in forecasts]))
    
    fig, axes = plt.subplots(total_levels, 1, figsize=(14, total_levels * 4))
    
    if total_levels == 1:
        axes = [axes]
    
    current_level = None
    level_index = 0
    
    for forecast in forecasts:
        if forecast['level'] != current_level:
            current_level = forecast['level']
            ax = axes[level_index]
            ax.set_title(f'Nível: {current_level}')
            level_index += 1
        
        ax.plot(test_dates, forecast['actual'], 'b-', label=f'Real {forecast["group"]}' if forecast['group'] == 'Total' else '')
        ax.plot(test_dates, forecast['forecast'], 'r--', label=f'Previsão {forecast["group"]}' if forecast['group'] == 'Total' else '')
    
    for ax in axes:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

    #