# %% [markdown]
# # Time Series Stationarity

# %% [markdown]
# ## What is Stationarity?

# %% [markdown]
# A time series is stationary if it does not exhibit any long term trends or obvious seasonality. 
# 
# It has:
# 
# - A constant variance through time
# - A constant mean through time
# - The statistical properties of the time series do not change

# %% [markdown]
# ## Visualise Data

# %% [markdown]
# (Data sourced from [Kaggle](https://www.kaggle.com/datasets/ashfakyeafi/air-passenger-data-for-time-series-analysis) with a CC0 licence)

# %%
# Import packages
import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

# %%
# Read in the data
data = pd.read_csv('AirPassengers.csv')

# %%
def plotting(title, data, x, y, x_label, y_label):
    """General function to plot the passenger data."""
    fig = px.line(data, x=data[x], y=data[y], labels={x: x_label, y: y_label})

    fig.update_layout(template="simple_white", font=dict(size=18),
                      title_text=title, width=650,
                      title_x=0.5, height=400)

    fig.show()

# %%
# Plot the airline passenger data
plotting(title='Airline Passengers', data=data, x='Month',
         y='#Passengers', x_label='Date', y_label='Passengers')

# %% [markdown]
# Is this time series stationary? No.
# 
# There is a clear increasing trend and the variance of fluctuations are also increasing in time.
# 
# To make the time series stationary, we need apply transformations to it.

# %% [markdown]
# ## Differencing

# %% [markdown]
# The most common transformation is differencing.

# %% [markdown]
# ![1*S6N2Aqb0IOdySLxar1hicQ.png](attachment:1*S6N2Aqb0IOdySLxar1hicQ.png)

# %% [markdown]
# Where d(t) is the difference at time t between the series at points y(t) and y(t-1).
# 

# %%
# Take the difference and plot it
data["Passenger_Diff"] = data["#Passengers"].diff()

plotting(title='Airline Passengers', data=data, x='Month', y='Passenger_Diff',
         x_label='Date', y_label='Passengers<br>Difference Transform')

# %% [markdown]
# Is the data now stationary? No.
# 
# The mean is now constant and is oscillating about zero. However, we can clearly see the variance is still increasing through time.

# %% [markdown]
# ## Logarithm Transform

# %% [markdown]
# To stabilise the variance, we apply the natural log transform.

# %%
# Take the log and plot it
data["Passenger_Log"] = np.log(data["#Passengers"])

plotting(title='Airline Passengers', data=data, x='Month',
         y='Passenger_Log', x_label='Date', y_label='Passenger<br>Log Transform')

# %% [markdown]
# The fluctuations are now on a consistent scale, but there is still a trend. Therefore, we now again have to apply the difference transform.

# %% [markdown]
# ## Logarithm and Differenc Transform

# %%
# Take the difference and log and plot it
data["Passenger_Diff_Log"] = data["Passenger_Log"].diff()

plotting(title='Airline Passengers', data=data, x='Month',
         y='Passenger_Diff_Log', x_label='Date', y_label='Passenger<br>Log and Difference')

# %% [markdown]
# Is the data now stationary? Yes!
# 
# As we can see, the mean and variance is now constant and has no long term trend.

# %% [markdown]
# ## Stationarity Test

# %% [markdown]
# There are more quantitative techniques to determine if the data is indeed stationary.
# 
# One such method is the Augmented Dickey-Fuller (ADF) test. This is a statistical hypothesis test where the null hypothesis is the series is non-stationary (also known as a unit root test).

# %%
# ADF test
def adf_test(series):
    """Using an ADF test to determine if a series is stationary"""
    test_results = adfuller(series)
    print('ADF Statistic: ', test_results[0])
    print('P-Value: ', test_results[1])
    print('Critical Values:')
    for thres, adf_stat in test_results[4].items():
        print('\t%s: %.2f' % (thres, adf_stat))


adf_test(data["Passenger_Diff_Log"][1:])

# %% [markdown]
# The ADF P-value (7.1%) is in-between the 5% and 10%, so depending on where you set your significance level we either reject or fail to reject the null hypothesis.

# %% [markdown]
# # Box-Cox Transform
# 

# %% [markdown]
# ## What is the Box-Cox Transform?

# %% [markdown]
# The Box-Cox transforms non-normal data to normal distribution like data.
# 
# Why do we need our time series data to resemble a normal distribution? Well, when fitting certain models, such as ARIMA, they use the maximum likelihood estimation (MLE) to determine their parameters. MLE by definition must fit against a certain distribution, which for most packages is the normal distribution.
# 
# The Box-Cox transformation is parameterised by λ (that takes real values from -5 to 5) and transforms the time series, y, as:

# %% [markdown]
# ![1*jf-zNg4Cu1bb3YIAjFActQ.png](attachment:1*jf-zNg4Cu1bb3YIAjFActQ.png)

# %% [markdown]
# We see that with λ=0 it is the natural log transform, however there are many others depending on the value λ.
# 
# For example, if λ=0 it is the square root transform, λ=1 there is no transform and λ=3 is the cubic transform.
# 
# The value λ is chosen by seeing which value best approximates the transformed data to the normal distribution. Luckily, in computing packages this is easily done for us!

# %% [markdown]
# ## Visualise Data
# 

# %%
# Import packages
import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy.stats import boxcox

# %%
# Read in the data
data = pd.read_csv('AirPassengers.csv')

# %%
def plotting(title, data, x, y, x_label, y_label):
    """General function to plot the passenger data."""
    fig = px.line(data, x=data[x], y=data[y], labels={x: x_label, y: y_label})

    fig.update_layout(template="simple_white", font=dict(size=18),
                      title_text=title, width=650,
                      title_x=0.5, height=400)

    fig.show()

# %%
# Plot the airline passenger data
plotting(title='Airline Passengers', data=data, x='Month',
         y='#Passengers', x_label='Date', y_label='Passengers')

# %% [markdown]
# The data is clearly not stationary as the mean and variance are both increasing with time. To stabilise the variance, we can use the Box-Cox transform like we discussed above.

# %% [markdown]
# ## Applying Box-Cox

# %%
# Apply box-cox transform and plot it
data['Passengers_box_cox'], lam = boxcox(data['#Passengers'])

plotting(title='Airline Passengers', data=data, x='Month', y='Passengers_box_cox',
         x_label='Date', y_label='Passengers<br>Box-Cox Transform', text=True, lam=lam)

# %% [markdown]
# Our variance is now stable and the fluctuations are on a consistent level! The optimal λ value is 0.148, which is near a perfect natural logarithmic transform but not quite. 

# %%


# %% [markdown]
# # Seasonality of Time Series
# 

# %% [markdown]
# ## Introduction

# %% [markdown]
# Seasonality is a crucial aspect of time-series analysis. As time-series are indexed forward in time, they are subject to seasonal fluctuations. For example, we expect ice cream sales to be higher in the summer months and lower in the winter months.

# %% [markdown]
# Seasonality can come in different time intervals such as days, weeks or months. The key for time-series analysis is to understand how the seasonality affects our series, therefore making us produce better forecasts for the future.

# %% [markdown]
# The easiest way to deal with seasonality is to remove it and make our **time-series stationary**, which is a requirement by most forecasting models. However, there are models such as SARIMA that model the seasonal affects for you.

# %% [markdown]
# ## Viewing Seasonality
# 

# %%
# Import packages
import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

# %%
# Read in the data
data = pd.read_csv('AirPassengers.csv')

# %%
def plotting(title, data, x, y, x_label, y_label):
    """General function to plot the passenger data."""
    fig = px.line(data, x=data[x], y=data[y], labels={x: x_label, y: y_label})

    fig.update_layout(template="simple_white", font=dict(size=18),
                      title_text=title, width=650,
                      title_x=0.5, height=400)

    fig.show()

# %%
# Plot the airline passenger data
plotting(title='Airline Passengers', data=data, x='Month',
         y='#Passengers', x_label='Date', y_label='Passengers')

# %% [markdown]
# The data is indexed by month and we can clearly see a yearly seasonal pattern where the number of passengers peaks in the summer months. There is also the overrall trend of the number of passengers increasing through time.

# %% [markdown]
# ## Removing Seasonality

# %% [markdown]
# We can remove seasonality in the data using seasonal differencing. This calculates the difference between the current value and its value in the previous season. The reason this is done is to make the time series stationary rendering its statistical properties constant through time. Seasonality causes the mean of the time series to be different when we are in a particular season. Hence, its statistical properties are not constant.

# %% [markdown]
# ![Screenshot%202023-09-02%20at%2011.42.25.png](attachment:Screenshot%202023-09-02%20at%2011.42.25.png)

# %% [markdown]
# Where d(t) is the differenced data point at time t, y(t) is the value of the series at t, y(t-m) is the value of the data point at the previous season and m is the length of one season. In our case m=12 as we have yearly seasonality.

# %%
# Take the seasonal difference and plot it
data["Passenger_Season_Diff"] = data["#Passengers"].diff(periods=12)

plotting(title='Airline Passengers', data=data, x='Month', y='Passenger_Season_Diff',
         x_label='Date', y_label='Passenger<br>Seasonal Difference')

# %% [markdown]
# The yearly seasonality has disappeared now, however we now observe some cycle. This is another common feature time series which is similar to seasonality but are typically on a longer timescale as observed here.

# %% [markdown]
# ## ADF Test

# %% [markdown]
# We can test that the resultant series is stationary using the Augmented Dickey-Fuller (ADF) test. The null hypothesis of this test is that the series is non-stationary. The statsmodels package provides a function for carrying out the ADF test:

# %%
def adf_test(series):
    """Using an ADF test to determine if a series is stationary"""
    test_results = adfuller(series)
    print('ADF Statistic: ', test_results[0])
    print('P-Value: ', test_results[1])
    print('Critical Values:')
    for thres, adf_stat in test_results[4].items():
        print('\t%s: %.2f' % (thres, adf_stat))


adf_test(data["Passenger_Season_Diff"][12:])

# %% [markdown]
# The P-Value is lower than the 5% and 10% threshold, but higher than the 1% threshold. Therefore, depending on your significance level we can either statistically confirm or deny that our series is stationary.

# %% [markdown]
# We can also carry out some further regular differencing (difference between adjacent values) to further reduce the P-Value. However, in this case I think the data is adequately stationary given it is below the 5% threshold.

# %%


# %% [markdown]
# # Time Series Decomposition
# 

# %% [markdown]
# ## Introduction

# %% [markdown]
# Understanding your time series is fundamental when trying to gain insight and finding the best model to produce future forecasts. Most time series can be broken up into different components to help diagnose it in a structured way providing a powerful analysis tool.

# %% [markdown]
# ## Time Series Components
# 

# %% [markdown]
# Time series are a combination of (mainly) three components: Trend, Seasonality and Residuals/Remainder. Let's break each of these down.
# 
# **Trend**: This is the overall motion of the series. It may be consistently increasing overtime, decreasing overtime or a combination of both.
# 
# **Seasonality**: Any regular seasonal pattern in the series. For example, ice cream sales are regularly higher in summer than winter.
# 
# **Residual/Remainder**: This is the bit that is left over after we take into account the trend and seasonality. It can also be thought of as just statistical noise.

# %% [markdown]
# ## Additive vs Multiplicative Model
# 

# %% [markdown]
# For an additive model we have:

# %% [markdown]
# ![Screenshot%202023-09-02%20at%2011.55.15.png](attachment:Screenshot%202023-09-02%20at%2011.55.15.png)

# %% [markdown]
# And for a multiplicative series:

# %% [markdown]
# ![Screenshot%202023-09-02%20at%2011.55.27.png](attachment:Screenshot%202023-09-02%20at%2011.55.27.png)

# %% [markdown]
# Where Y is the series, T is the trend, S is the seasonality and R is the residual component.

# %% [markdown]
# The additive model is most appropriate when the size of the series’ variations are on a consistent numerical scale. On the other hand, the multiplicative model is when the series’ fluctuations are on a relative scale.
# 
# For example, if the ice cream sales are higher in summer by 1,000 every year, then the model is additive. If the sales are higher by a consistent 20% every summer, but the absolute number of sales are changing, then the model is multiplicative. 

# %% [markdown]
# It is possible to convert a multiplicative model to an additive one by simply taking the log transfrom or the Box-Cox transform:

# %% [markdown]
# ![Screenshot%202023-09-02%20at%2011.59.57.png](attachment:Screenshot%202023-09-02%20at%2011.59.57.png)

# %% [markdown]
# ## How is Decomposition Done?

# %% [markdown]
# There are multiple algorithms and methods to decompose the time series into the three components. I want to go over the classical approach as this is frequently used and is quite intuitive.
# 
# - Compute the trend component, T, using a moving/rolling average.
# 
# - De-trend the series, Y-T for additive model and Y/T for multiplicative model.
# 
# - Compute the seasonal component, S, by taking the average of the de-trended series for each season.
# - The residual component, R, is calculated as: R = Y-T-R for additive model and R = Y/(TR) for multiplicative model.
# 
# There are also several other methods available for decomposition such as STL, X11 and SEATS. These are advanced methods and add to the basic approach from the classical method and improve upon its shortcomings.

# %% [markdown]
# ## Python Example

# %%
# Import packaged
import plotly.express as px
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# %%

# Read in the data
data = pd.read_csv('AirPassengers.csv', index_col=0)
data.index = pd.to_datetime(data.index)

# Plot the data
fig = px.line(data, x=data.index, y='#Passengers',
              labels=({'#Passengers': 'Passengers', 'Month': 'Date'}))

fig.update_layout(template="simple_white", font=dict(size=18),
                  title_text='Airline Passengers', width=650, title_x=0.5, height=400)

fig.show()

# %% [markdown]
# From this plot we observe an increasing trend and a yearly seasonality. Notice that the size of the fluctuations are increasing through time, therefore we have a multiplicative model.
# 

# %% [markdown]
# ### Muliplicative 

# %%
# Plot the decomposition for multiplicative series
data.rename(columns={'#Passengers': 'Multiplicative Decomposition'}, inplace=True)
decomposition_plot_multi = seasonal_decompose(data['Multiplicative Decomposition'],
                                              model='multiplicative')
decomposition_plot_multi.plot()
plt.show()

# %% [markdown]
# From the plot above we can see that the function has indeed successfully captured the three components.

# %% [markdown]
# ### Additive 

# %% [markdown]
# We can convert our series to an additive model by stabilising the variance using the Box-Cox transform by applying the boxcox Scipy function:

# %%
# Apply boxcox to acquire additive model
data['Additive Decomposition'], lam = boxcox(data['Multiplicative Decomposition'])

# Plot the decomposition for additive series
decomposition_plot_add = seasonal_decompose(data['Additive Decomposition'],
                                            model='additive')
decomposition_plot_add.plot()
plt.show()

# %% [markdown]
# Again, the function seems to have captured the three components well. Interestingly, we see the residuals having a higher volatility in the earlier and later years. This may be something to take into account when building a forecasting model for this series.

# %%


# %% [markdown]
# # Autocorrelation
# 
# 

# %% [markdown]
# ## Introduction

# %% [markdown]
# In time series analysis we often make inferences about the past to produce forecasts about the future. In order for this process to be successful, we must diagnose our time series thoroughly.
# 
# One such diagnosis method is autocorrelation. This helps us detect certain features in our series to enable us to choose the most optimal forecasting model for our data.

# %% [markdown]
# ## What is Autocorrelation?
# 

# %% [markdown]
# Autocorrelation is just the correlation of the data with itself. So, instead of measuring the correlation between two random variables, we are measuring the correlation between a random variable against itself. Hence, why it is called auto-correlation.
# 
# Correlation is how strongly two variables are related to each other. If the value is 1, the variables are perfectly positively correlated, -1 they are perfectly negatively correlated and 0 there is no correlation.
# 
# 
# For time-series, the autocorrelation is the correlation of that time series at two different points in time (also known as lags). In other words, we are measuring the time series against some lagged version of itself.

# %% [markdown]
# ![Screenshot%202023-09-10%20at%2020.41.29.png](attachment:Screenshot%202023-09-10%20at%2020.41.29.png)

# %% [markdown]
# Where N is the length of the time series y and k is the specifie lag of the time series. So, when calculating r_1 we are computing the correlation between y_t and y_{t-1}.
# 
# The autocorrelation between y_t and y_t would be 1 as they are identical.

# %% [markdown]
# ## Why is it Useful?
# 

# %% [markdown]
# As stated above, we use autocorrelation to measure the correlation of a time series with a lagged version of itself. This computation allows us to gain some interesting insight into the characteristics of our series:
# 
# - Seasonality: Lets say we find the correlation at certain lag multiples is in general higher than others. This means we have some seasonal component in our data. For example, if we have daily data and we find that every multiple of 7 lag term is higher than others, we probably have some weekly seasonality.
# 
# - Trend: If the correlation for recent lags is higher and slowly decreases as the lags increase, then there is some trend in our data. Therefore, we would need to carry out some differencing to render the time series stationary.

# %% [markdown]
# ## Python Example

# %%
# Import packaged
import plotly.express as px
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# %%
# Read in the data
data = pd.read_csv('AirPassengers.csv', index_col=0)
data.index = pd.to_datetime(data.index)

# Plot the data
fig = px.line(data, x=data.index, y='#Passengers',
              labels=({'#Passengers': 'Passengers', 'Month': 'Date'}))

fig.update_layout(template="simple_white", font=dict(size=18),
                  title_text='Airline Passengers', width=650, title_x=0.5, height=400)

fig.show()

# %%
# Plot autocorrelation
plt.rc("figure", figsize=(8,4))
plot_acf(data['#Passengers'], lags=48)
plt.ylim(0,1)
plt.xlabel('Lags', fontsize=18)
plt.ylabel('Correlation', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Autocorrelation Plot', fontsize=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# We observe the following:
# 
# - There is a clear cyclical pattern in the lags every multiple of 12. As our data is indexed by month, we therefore have a yearly seasonality in our data.
# - The strength of correlation is generally and slowly decreasing as the lags increase. This points to a trend in our data and it needs to be differenced to make it stationary when modelling.
# 
# The blue region signifies which lags are statistically significant. Therefore, when building a forecast model for this data, the next month forecast should probably only consider ~15 of the previous values due to their statistical significance.

# %% [markdown]
# # Partial Autocorrelation
# 
# 

# %% [markdown]
# ## What is Partial Autocorrelation?
# 

# %% [markdown]
# We can begin by explaining partial correlation. This is the correlation between two random variables whilst controlling the effect of another (orm more) random variable that affects the original variables we are correlating.
# 
# Lets say we have three random variables of X, Y and Z. The partial correlation between X and Y, excluding the effects of Z, is mathematically:

# %% [markdown]
# ![Screenshot%202023-09-10%20at%2020.52.43.png](attachment:Screenshot%202023-09-10%20at%2020.52.43.png)

# %% [markdown]
# Where r is the correlation coefficient that ranges between -1 and 1.
# 
# Partial autocorrelation is then simply just the partial correlation of a time series at two different states in time. Taking it one step further, it is the correlation between the time series at two different lags not considering the effect of any intermediate lags. For example, the partial autocorrelation for a lag of 2 is only the correlation that lag 1 didn’t explain.

# %% [markdown]
# ## Why is it Useful?
# 

# %% [markdown]
# Unlike autocorrelation, partial autocorrelation hasn’t got as my uses for time series analysis. However, its main and very important impact comes in when building forecasting models.
# 
# The PACF is used to estimate the number/order of autoregressive components when fitting Autoregressive, ARMA or ARIMA models as defined by the Box-Jenkins procedure. These models are probably the most used and often provide the best results when training a forecasting model.

# %% [markdown]
# ## Python Example

# %%
# Import packaged
import plotly.express as px
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

# %%
# Read in the data
data = pd.read_csv('AirPassengers.csv', index_col=0)
data.index = pd.to_datetime(data.index)

# Plot the data
fig = px.line(data, x=data.index, y='#Passengers',
              labels=({'#Passengers': 'Passengers', 'Month': 'Date'}))

fig.update_layout(template="simple_white", font=dict(size=18),
                  title_text='Airline Passengers', width=650, title_x=0.5, height=400)

fig.show()

# %%
# Plot partial autocorrelation
plt.rc("figure", figsize=(11,5))
plot_pacf(data['#Passengers'], method='ywm')
plt.xlabel('Lags', fontsize=18)
plt.ylabel('Correlation', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Partial Autocorrelation Plot', fontsize=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# We see that lag 1 is highly correlated and there are other highly correlated lags later in time which are probably due to seasonal affects.
# 
# However, from this plot it is quite unclear how many autoregressors we would choose if we were building a forecasting model. Therefore, it is often recommended to simply carry out a grid-search over the possible parameters using modelling packages such as auto arima.
# 

# %% [markdown]
# The blue region is where lags are no longer statistically significant. We typically choose the number autoregressors by seeing how many of them are above the blue region.

# %% [markdown]
# # How To Analyse Your Time Series Model Using Residuals
# 
# 

# %% [markdown]
# ## Intro
# 

# %% [markdown]
# Being able to analyse your time series model is essential to diagnose its performance. One such way to do this is through the residuals of the fitted model. In this post, we will go over what residuals are and how they can be used to improve your model along with an example in Python.
# 

# %% [markdown]
# ## What are Residuals?
# 
# 

# %% [markdown]
# In time series analysis, residuals, r, are the difference between the fitted values, ŷ, and the actual values, y:

# %% [markdown]
# ![Screenshot%202023-10-15%20at%2022.43.49.png](attachment:Screenshot%202023-10-15%20at%2022.43.49.png)

# %% [markdown]
# It is important to state the difference between residuals and errors. The error is the difference between the actual and forecasted values. However, the residuals, as shown above, are the difference from the actual the fitted values. These fitted values are the predictions the model made to the training data whilst fitting to it. As the model knows the values of all observations, it is no longer technically a forecast but rather a fitted value.

# %% [markdown]
# ## Residual Analysis
# 
# 

# %% [markdown]
# We can use the residuals to analyse how well our model has captured the characteristics of the data. In general, the residuals should:
# 
# - Show very little or no autocorrelation or partial autocorrelation. If they have any form of correlation, then the model has missed some information that’s in the data. We can use the Ljung–Box statistical test and a correlogram to determine if the residuals are indeed correlated.
# - The mean of the residuals should be zero, otherwise the forecast will be biased. In reality, this is quite easy to adjust for by simply adding or subtracting the bias from the forecasts.

# %% [markdown]
# ##  In Python

# %% [markdown]
# ### Fitting a Holt Winters’ Model

# %% [markdown]
# For this short walkthrough, we will fit the exponential smoothing Holt Winters’ model to the famous US airline passenger dataset.

# %%
# Import packages
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])

# Split train and test
train = data.iloc[:-int(len(data) * 0.2)]
test = data.iloc[-int(len(data) * 0.2):]


def plot_func(forecast: list[float],
              title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Month'], y=train['#Passengers'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Month'], y=test['#Passengers'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Month'], y=forecast, name='Forecast'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=700, title_x=0.5, height=400, xaxis_title='Date',
                      yaxis_title='Passenger Volume')

    return fig.show()


# Fit Holt Winters model and get forecasts
model = ExponentialSmoothing(train['#Passengers'], trend='mul', seasonal='mul', seasonal_periods=12)\
    .fit(optimized=True)
forecasts = model.forecast(len(test))

# Plot the forecasts
plot_func(forecasts,  "Holt Winters Forecast")

# %% [markdown]
# The forecast from this model looks pretty good. 

# %%
# Appending residuals and fitted values to the train dataframe
train['fittedvalues'] = model.fittedvalues
train['residuals'] = model.resid
print(train)

# %% [markdown]
# ### Residual Correlation

# %% [markdown]
# The correlation of the residuals can be computed by plotting their autocorrelation and partial autocorrelation function.

# %%
# Import packages
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt

# Plot ACF and PACF
fig, ax = plt.subplots(1,2,figsize=(8,3))
plot_acf(train['residuals'], lags=30, ax=ax[0])
ax[0].set_xlabel('Lags')
plot_pacf(train['residuals'], lags=30, ax=ax[1])
ax[1].set_xlabel('Lags')
plt.tight_layout()
plt.show()

# %% [markdown]
# Majority of the correlations are within the non-statistically significant blue region, which would signify that the residuals are not correlated. However, you may notice that there is some recurring pattern in the correlations. This would convey that there is some seasonal component that the model may have not fully accounted for.

# %% [markdown]
# ### Ljung-Box Test
# 

# %% [markdown]
# A more quantitive way to determine if the residuals are correlated is to carry out the Ljung–Box statistical test:

# %% [markdown]
# H_0: Residuals are independently distributed.
# 
# H_1: Residuals are not independently distributed and exhibit serial correlation.

# %%
# Import packages
from statsmodels.stats.diagnostic import acorr_ljungbox

# Carry out Ljung-Box test
print(acorr_ljungbox(train['residuals'], return_df=True))


# %% [markdown]
# This shows the p-values of the first 10 lags. They are all below the significance level of 0.05, therefore we reject the null hypothesis of no autocorrelation. Thus, there is correlation present in our residuals that we need to revisit when re-fitting the model.

# %% [markdown]
# ### Histogram of Residuals
# 

# %% [markdown]
# A histogram of the residuals will determine if they have a mean of zero and are symmetric (no bias):

# %%
# Import plotly
import plotly.express as px

# Plot histogram of the residuals
fig = px.histogram(train, x="residuals")
fig.update_layout(template="simple_white", font=dict(size=18),
                  title_text='Distribution of Residuals',
                  width=700, title_x=0.5, height=400,
                  xaxis_title='Residuals', yaxis_title='Count')
fig.show()

# Mean of residuals
print(train['residuals'].mean())

# %% [markdown]
# In this case the residuals are mostly distributed around zero with a mean of -0.023 and maybe even slightly negatively biased. This suggests that we probably don’t need to provide an offset for the computed forecasts.

# %%


# %% [markdown]
# # Crossvalidation for Time Series
# 
# 

# %% [markdown]
# ## Intro
# 

# %% [markdown]
# Cross-validation is a staple process when building any statistical or machine learning model and is ubiquitous in data science. However, for the more niche area of time series analysis and forecasting, it is very easy to incorrectly carry out cross-validation.

# %% [markdown]
# ## What Is Cross-Validation?
# 
# 

# %% [markdown]
# Cross-validation is a method to determine the best performing model and parameters through training and testing the model on different portions of the data. The most common and basic approach is the classic train-test split. This is where we split our data into a training set that is used to fit our model and then evaluated it on the test set.
# 
# This idea can be taken one step further by carrying out the train-test split numerous times by varying the data we train and test on. This process is cross-validation as we are using every row of data for both training and evaluation to ensure we choose the most robust model over all the possible available data.

# %%
# Import packages
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import KFold


def plot_cross_val(n_splits: int,
                   splitter_func,
                   df: pd.DataFrame,
                   title_text: str) -> None:
  
    """Function to plot the cross validation of various
    sklearn splitter objects."""

    split = 1
    plot_data = []

    for train_index, valid_index in splitter_func(n_splits=n_splits).split(df):
        plot_data.append([train_index, 'Train', f'{split}'])
        plot_data.append([valid_index, 'Test', f'{split}'])
        split += 1

    plot_df = pd.DataFrame(plot_data,
                           columns=['Index', 'Dataset', 'Split'])\
                           .explode('Index')

    fig = go.Figure()
    for _, group in plot_df.groupby('Split'):
        fig.add_trace(go.Scatter(x=group['Index'].loc[group['Dataset'] == 'Train'],
                                 y=group['Split'].loc[group['Dataset'] == 'Train'],
                                 name='Train',
                                 line=dict(color="blue", width=10)
                                 ))
        fig.add_trace(go.Scatter(x=group['Index'].loc[group['Dataset'] == 'Test'],
                                 y=group['Split'].loc[group['Dataset'] == 'Test'],
                                 name='Test',
                                 line=dict(color="goldenrod", width=10)
                                 ))

    fig.update_layout(template="simple_white", font=dict(size=20),
                      title_text=title_text, title_x=0.5, width=850,
                      height=450, xaxis_title='Index', yaxis_title='Split')

    legend_names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in legend_names) else legend_names.add(trace.name))

    return fig.show()
  
# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])

# Plot the cross validation
plot_cross_val(n_splits=5,
              splitter_func=KFold,
              df=data,
              title_text='Cross-Validation')

# %% [markdown]
# ## Time Series Cross Validation
# 
# 

# %% [markdown]
# The above cross-validation is not an effective or valid strategy on forecasting models due to their temporal dependency. For time series, we always predict into the future. However, in the above approach we will be training on data that is further in time than the evaluation test data. This is data leakage and should be avoided at all costs.
# 
# To overcome this problem, we need to ensure the test set always has a higher index (the index is usually time for time series data) than the training set. This means our test is always in the future compared to the data our model is fitted on.
# 

# %%
# Import packages
from sklearn.model_selection import TimeSeriesSplit

# Plot the time series cross validation splits
plot_cross_val(n_splits=5,
               splitter_func=TimeSeriesSplit,
               df=data,
               title_text='Time Series Cross-Validation')

# %% [markdown]
# ## Hyperparameter Tuning
# 

# %% [markdown]
# Cross-validation is frequently used in collaboration with hyperparameter tuning to determine the optimal hyperparameter values for a model. Let’s quickly go over an example of this process, for a forecasting model, in Python.

# %%
# Import packages
import plotly.express as px


def plot_time_series(df: pd.DataFrame) -> None:
    """General function to plot the passenger data."""
    
    fig = px.line(df, x='Month', y='#Passengers',
                  labels={'Month': 'Date', '#Passengers': 'Passengers'})
                  
    fig.update_layout(template="simple_white", font=dict(size=18),
                      title_text='Airline Passengers', width=650,
                      title_x=0.5, height=400)

    return fig.show()
    
    
# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])

# Plot the time series
plot_time_series(df=data)

# %% [markdown]
# The data has a clear trend and high seasonality. A suitable model for this time series would be the Holt Winters exponential smoothing model that incorporates both trend and seasonality components

# %%
# Import packages
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def hyperparameter_tuning_season_cv(n_splits: int,
                                    gammas: list[float],
                                    df: pd.DataFrame) -> pd.DataFrame:                                   
    """Function to carry out cross-validation hyperparameter tuning
    for the seasonal parameter in a Holt Winters' model. """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    error_list = []

    for gamma in gammas:
    
        errors = []
        
        for train_index, valid_index in tscv.split(df):
            train, valid = df.iloc[train_index], df.iloc[valid_index]
            
            model = ExponentialSmoothing(train['#Passengers'], trend='mul',
                                         seasonal='mul', seasonal_periods=12) \
                .fit(smoothing_seasonal=gamma)
                
            forecasts = model.forecast(len(valid))
            errors.append(mean_absolute_percentage_error(valid['#Passengers'], forecasts))

        error_list.append([gamma, sum(errors) / len(errors)])

    return pd.DataFrame(error_list, columns=['Gamma', 'MAPE'])
    
 
def plot_error_cv(df: pd.DataFrame,
                  title: str) -> None:                  
    """Bar chart to plot the errors from the different
    hyperparameters."""

    fig = px.bar(df, x='Gamma', y='MAPE')
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=800, title_x=0.5, height=400)

    return fig.show()
    
    
# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])

# Carry out cv for hyperparameter tuning for the seasonal parameter
error_df = hyperparameter_tuning_season_cv(df=data,
                                         n_splits=4,
                                         gammas=list(np.arange(0, 1.1, 0.1)))

# Plot the tuning results
plot_error_cv(df=error_df, title='Hyperparameter Results')

# %% [markdown]
# As we can see, it appears the optimal value of the smoothing_seasonal hyperparameter is 0.7.
# 

# %%


# %% [markdown]
# # How To Forecast Time-Series Using Autoregression
# 

# %% [markdown]
# ## Intro
# 

# %% [markdown]
# In my previous posts we have covered autoregession (AR) and moving-average (MA) models. However, do you know what is better than these two models? A single model that combines them!
# 
# Autoregressive Integrated Moving Average better known as ARIMA, is probably the most used time series forecasting model and is combination of the individual aforementioned models.
# 
# In this article, I want to dive into the theory and framework behind the ARIMA model. Then, we will go through a simple Python walkthrough in carrying out forecast with ARIMA using the statsmodels package!
# 

# %% [markdown]
# ## What Is Moving Average Mode?
# 
# 

# %% [markdown]
# ### Overview

# %% [markdown]
# As stated above, ARIMA stands for AutoRegressive Integrated Moving Average is basically just a combination of the three (in reality two) components:
# 

# %% [markdown]
# **AutoRegressive (AR):**

# %% [markdown]
# This is just autoregression, where we forecast future values using a linear combination of the previously observed values:

# %% [markdown]
# ![Screenshot%202023-10-22%20at%2012.45.07.png](attachment:Screenshot%202023-10-22%20at%2012.45.07.png)

# %% [markdown]
# Here y is the time series we are forecasting at multiple time steps, ϕ are the coefficients of the lags, ε is the error term (often normally distributed) and p is the number of lagged components, also known as the order.

# %% [markdown]
# **Integrated (I):**

# %% [markdown]
# The middle part of the ARIMA model is named integrated. This is the number (order d) of differencing required to make the time series stationary.
# 
# Stationarity is where the time series has a constant mean and variance, meaning the statistical properties of the series does not change through time. Differencing de-trends a time series and tends to make the mean constant. You can apply differencing several times, but often the series is sufficiently stationary after a single differencing step.
# 
# It is important to note, that this integrated part only makes the mean constant. We need to apply another transform such as the logarithmic and Box-Cox transform to generate a constant variance (more on this later).

# %% [markdown]
# **Moving Average (MA):**

# %% [markdown]
# The last component is the moving average where you forecast using past forecast errors instead of the actual observed values:

# %% [markdown]
# ![Screenshot%202023-10-22%20at%2015.02.08.png](attachment:Screenshot%202023-10-22%20at%2015.02.08.png)

# %% [markdown]
# Here y is the time series we are forecasting at multiple time steps, μ is the mean, θ are the coefficients of the lagged forecast errors, ε are the forecast error terms and q is the number of lagged error components.
# 

# %% [markdown]
# **Final Form:**

# %% [markdown]
# Combining all these components together, we can write the full model as:
# 

# %% [markdown]
# ![Screenshot%202023-10-22%20at%2015.13.58.png](attachment:Screenshot%202023-10-22%20at%2015.13.58.png)

# %% [markdown]
# Where y’ refers to the differenced version of the time series.
# 
# This is the full ARIMA equation and is just a linear summation of the three components. The model is usefully written in a short-hand way as ARIMA(p, d, q) where p, d and q refer to the order of autoregressors, differencing and moving-averages components respectively.

# %% [markdown]
# ### Requirements

# %% [markdown]
# As we touched upon earlier, the differencing component is there to help make the time series stationary. This is because the ARIMA model requires the data to be stationary for it to adequately model it. The mean is stabilised through differencing and the variance can be stabilised through the Box-Cox transform as we mentioned above.
# 

# %% [markdown]
# ### Order Selection

# %% [markdown]
# One of the preprocessing steps is to determine the optimal orders (p, d, q) of our ARIMA model. The simplest one is the order of differencing d as this can be verified by carrying out a statistical test for stationarity. The most popular one is the Augmented Dickey-Fuller (ADF), where the null hypothesis is that the time series is not stationary.
# 
# The autoregressive and moving-average orders (p,q) can be deduced by analysing the partial autocorrelation function (PACF) and autocorrelation function respectively. The gist of of this method is to plot a correlogram of the various lags/forecast errors of the time series to determine which are statistically significant. If this seems arbitrary at the moment don’t worry, in the Python tutorial later we will walkthrough this process.
# 
# However, a more thorough technique is to simply iterate over all the possible combinations of orders and choose the model with the best score against a metric such as Akaike’s Information Criterion (AIC) or Bayesian Information Criterion (BIC). This is analogous to regular hyperparameter tuning and definitely the more robust method, but is more computational expensive of course.
# 

# %% [markdown]
# ### Estimation

# %% [markdown]
# After choosing our orders, we then need to find their optimal corresponding coefficients. This where the need for stationarity comes in. As declared above, a stationary time series has constant statistical properties such as mean and variance. Therefore, all the data points are part of the same probability distribution, which makes fitting our model easier. Furthermore, forecasts are treated as random variables and will now belong to the same probability distribution as the newly generated stationary time series. Overall, it helps make the data in the future be somewhat like the past.
# 
# As the stationary data belongs to some distribution (frequently the normal distribution), we can estimate the coefficients using Maximum Likelihood Estimation (MLE). MLE deduces the optimal values of the coefficients that produce the highest probability of obtaining that data. The MLE for normally distributed data, is the same result as carrying ordinary least squares. Therefore, least squares is also frequently used for this exact reason.

# %% [markdown]
# ## Python Walkthrough
# 
# 

# %% [markdown]
# ### Data

# %%
# Import packages
import plotly.express as px
import pandas as pd

# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])


def plot_passenger_volumes(df: pd.DataFrame,
                           y: str) -> None:
    """General function to plot the passenger data."""

    fig = px.line(df, x='Month', y=y, labels={'Month': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='Airline Passengers',
                      width=650, title_x=0.5, height=400)

    return fig.show()


# Plot the airline passenger data
plot_passenger_volumes(df=data, y='#Passengers')

# %% [markdown]
# The data is not stationary as there is a strong positive trend and the yearly seasonality fluctuations are increasing through time, hence the variance is increasing. For this modelling task we will be using the statsmodel package, which handily carries out differencing for us and produces a constant mean. However, we still need to apply the Box-Cox transform to retrieve a stabilised variance:

# %%
# Import packages 
from scipy.stats import boxcox

# Make the target variance stationary
data['Passengers_Boxcox'], lam = boxcox(data['#Passengers'])

# Plot the box-cox passenger data
plot_passenger_volumes(df=data, y='Passengers_Boxcox')

# %% [markdown]
# The data now appears to be stationary. We could have made it further stationary by carrying out second order differencing or seasonal differencing, however I think it is satisfactory here.

# %% [markdown]
# ### Modelling

# %% [markdown]
# In the previous sections, I mentioned how you can find the autoregressive and moving-average orders by plotting the autocorrelation and partial autocorrelation functions. Let’s show an example of how you can do it here:

# %%
# Import packages
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Difference the data
data["Passenger_diff"] = data["Passengers_Boxcox"].diff()
data.dropna(inplace=True)

# Plot acf and pacf
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), dpi=80)
plot_acf(data['Passenger_diff'])
plot_pacf(data['Passenger_diff'], method='ywm')
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.show()

# %% [markdown]
# The blue region signifies where the points are no longer statistically significant and from the plot we see the last lag that is statistically significant for both plot is ~12th. Therefore, we would take the order of p and q to be 12.
# 
# Now, let’s fit the model using the ARIMA function and generate the forecasts:

# %%
# Import packages
from statsmodels.tsa.arima.model import ARIMA
from scipy.special import inv_boxcox

# Split train and test
train = data.iloc[:-int(len(data) * 0.2)]
test = data.iloc[-int(len(data) * 0.2):]

# Build ARIMA model and inverse the boxcox
model = ARIMA(train['Passengers_Boxcox'], order=(12, 1, 12)).fit()
boxcox_forecasts = model.forecast(len(test))
forecasts = inv_boxcox(boxcox_forecasts, lam)

# %% [markdown]
# ### Results

# %% [markdown]
# The forecasts produced from this fitted model is for the differenced and Box-Cox transformed time series that we produced earlier. Therefore, we have to un-difference and apply the inverse Box-Cox transform to the predictions to acquire the actual airline passenger forecasted volumes:

# %%
# Import packages
import plotly.graph_objects as go

def plot_forecasts(forecasts: list[float], title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Month'], y=train['#Passengers'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Month'], y=test['#Passengers'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Month'], y=forecasts, name='Forecast'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=650, title_x=0.5, height=400, xaxis_title='Date',
                      yaxis_title='Passenger Volume')

    return fig.show()


# Plot the forecasts
plot_forecasts(forecasts, 'ARIMA')

# %%


# %% [markdown]
# # How To Forecast Time-Series Using SARIMA
# 

# %% [markdown]
# ## Intro
# 

# %% [markdown]
# In one of my previous posts we covered probably the most famous forecasting model, Autoregressive Integrated Moving Average better known as ARIMA. However, one disadvantage of this model is that it is lacking awareness of any seasonality. This is where the Seasonal Autoregressive Integrated Moving Average, or SARIMA, model comes in. In this post, we will take a deep dive into the theory and main ideas behind the SARIMA model and how to implement it in Python.

# %% [markdown]
# ## What Is SARIMA
# 
# 

# %% [markdown]
# ### Overview

# %% [markdown]
# SARIMA is an extension of the regular ARIMA model that adds a seasonality component to the model. This allows us to better capture seasonal affects that the regular ARIMA model does not permit.

# %% [markdown]
# ### Theory

# %% [markdown]
# The classic ARIMA model has three components: Autoregressive, Integrated (differencing), and Moving-Average. These are then linearly combined to form the model:

# %% [markdown]
# ![Screenshot%202023-11-08%20at%2020.55.07.png](attachment:Screenshot%202023-11-08%20at%2020.55.07.png)

# %% [markdown]
# Where:
# - y’: differenced time series, the number of differencing applied is noted as d
# - ϕ: coefficients of the autoregressive components (lags)
# - p: number of autoregressive components
# - ε: forecast error terms, the moving-average components
# - θ: coefficients of the lagged forecast errors
# - q: number of lagged error components
# 
# The model is often compactly written ARIMA(p, d, q) where p, d, and q refer to the order of autoregressors, differencing and moving-average components respectively.

# %% [markdown]
# SARIMA adds a seasonality component to each factor of the ARIMA equation to produce SARIMA(p, d, q)(P, D, Q)m:

# %% [markdown]
# ![Screenshot%202023-11-08%20at%2020.56.12.png](attachment:Screenshot%202023-11-08%20at%2020.56.12.png)

# %% [markdown]
# Where:
# - y’: differenced time series, through both regular, d, and seasonal, D, differencing
# - P: number of seasonal auto-regressors
# - ω: coefficients of the seasonal autoregressive components
# - Q: number of seasonal moving-average components
# - η: coefficients of the seasonal forecast errors
# - m: length of season

# %% [markdown]
# ### Requirements

# %% [markdown]
# Like the original ARIMA model, the SARIMA model needs to have stationary data to model and forecast the time series. A stationarity time series does not exhibit any long-term trend or clear seasonality, its statistical properties, such as mean and variance, remain constant over time.
# 
# To produce a stationary time series we need to stabilize the mean and variance. The mean can be stabilized through differencing and the number of differencing applied is d or D in the case of seasonal differencing. The variance can be stabilized through transformations such as the logarithmic and Box-Cox transform, this makes the seasonal fluctuations occur on a similar level every season.

# %% [markdown]
# ### Order Selection

# %% [markdown]
# After the time series is stationary, we then need to deduce the best orders, (p, d, q) and (P, D, Q)m, for our model. The simplest one to calculate is the seasonal, D, and regular differencing, d. This can be deduced through the Augmented Dickey-Fuller (ADF) statistical test that deduces whether a time series is stationary or not.
# 
# The autoregressive and moving-average (forecast errors) orders (p, q, P, Q) can be computed by analyzing the partial autocorrelation function (PACF) and autocorrelation function respectively. The idea behind this technique is to plot a correlogram of the autoregressors and moving-average value and deduce which ones are statistically significant. The significant ones indicate that they have a substantial impact on the forecast.
# 
# These correlograms will also allow us to observe the seasonal pattern if any, as we may see peaks at certain multiple lags. For example, a SARIMA(0,0,0)(1,0,0)4 will show exponential decay in the lags for the ACF but a significant spike at lag 4 in the PACF. If the data is indexed by month, then this is would be an example of quarterly seasonality.

# %% [markdown]
# ### Estimation

# %% [markdown]
# The final step is to compute the corresponding coefficients for these orders. The most common method is to use Maximum Likelihood Estimation (MLE) which estimates the coefficients against some assumed probability distribution, typically normal, to calculate which coefficient is the most likely to generate that data. As the time series is stationary and has constant statistical properties, we can say that it belongs to some probability distribution allowing us to use MLE. This is why stationarity is the key requirement for SARIMA.

# %% [markdown]
# ## Python Walkthrough
# 
# 

# %% [markdown]
# ### Data

# %%
# Import packages
import plotly.express as px
import pandas as pd

# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])


def plot_passenger_volumes(df: pd.DataFrame,
                           y: str) -> None:
    """General function to plot the passenger data."""

    fig = px.line(df, x='Month', y=y, labels={'Month': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='Airline Passengers',
                      width=650, title_x=0.5, height=400)

    return fig.show()


# Plot the airline passenger data
plot_passenger_volumes(df=data, y='#Passengers')

# %% [markdown]
# There is an obvious trend and seasonality, so the data is not stationary as the mean and variance is changing over time. Therefore, we need to apply differencing and the Box-Cox transform to make our series stationary as required by SARIMA:

# %%
# Import packages
from scipy.stats import boxcox

# Make the data stationary
data['Passengers_Boxcox'], lam = boxcox(data['#Passengers'])
data["Passenger_diff"] = data["Passengers_Boxcox"].diff()
data.dropna(inplace=True)

# Plot the stationary passenger data
plot_passenger_volumes(df=data, y='Passenger_diff')

# %% [markdown]
# The data now looks sufficiently stationary.
# 

# %% [markdown]
# ### Modelling

# %% [markdown]
# We will now use the ACF and PACF correlograms to deduce the orders for the autoregressive and moving-average components:

# %%
# Import packages
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Plot acf and pacf
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=80)
plot_acf(data['Passenger_diff'])
plot_pacf(data['Passenger_diff'], method='ywm')
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.show()

# %% [markdown]
# We already observed that our series yearly seasonality, m=12, but the above plots confirm this as we have large spikes at the 12th lags. The lags are also significant to around ~10th lag for both plots. Overall this indicates that a SARIMA(10, 1, 10)(1, 1, 1)12 model should be suitable.
# 
# Now, let’s fit the model using the ARIMA class from statsmodels and generate the forecasts. Luckily, this class carries out differencing for us, so we only need to pass the Box-Cox transformed time series:

# %%
# Import packages
from scipy.special import inv_boxcox
from statsmodels.tsa.arima.model import ARIMA

# Split train and test
train = data.iloc[:-int(len(data) * 0.2)]
test = data.iloc[-int(len(data) * 0.2):]

# Build ARIMA model
model = ARIMA(train['Passengers_Boxcox'], order=(10, 1, 10),
              seasonal_order=(1, 1, 1, 12)).fit()
boxcox_forecasts = model.forecast(len(test))
forecasts = inv_boxcox(boxcox_forecasts, lam)

# %% [markdown]
# ### Results

# %% [markdown]
# Finally, we will plot the forecasts:

# %%
# Import packages
import plotly.graph_objects as go

def plot_forecasts(forecasts: list[float], title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Month'], y=train['#Passengers'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Month'], y=test['#Passengers'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Month'], y=forecasts, name='Forecast'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=650, title_x=0.5, height=400, xaxis_title='Date',
                      yaxis_title='Passengers')

    return fig.show()


# Plot the forecasts
plot_forecasts(forecasts, 'SARIMA')

# %% [markdown]
# The SARIMA forecasts seemed to have done quite well!
# 

# %% [markdown]
# # Fourier Series & Harmonic Regression For Time Series Analysis
# 

# %% [markdown]
# ## Background
# 

# %% [markdown]
# When we want to model seasonality in our time series we often turn to the SARIMA model. This adds seasonality components to the ARIMA model by finding autoregressors and moving-averages at certain specific lag indexes. For example, monthly data with yearly seasonality will fit autoregressors and moving averages at multiples of 12.
# 
# However, what if we have daily data with a yearly seasonality of 365.25 days? Or even weekly data with a seasonality of 52.14?
# 
# Unfortunately, SARIMA can’t handle this as it is non-integer and also struggles computationally due to the memory required to find patterns in 365 data points each season.
# 
# So, what do we do?

# %% [markdown]
# ## What are Fourier Series?
# 
# 

# %% [markdown]
# ### Overview

# %% [markdown]
# Fourier series is one of the most interesting discoveries in mathematics which states that:
# 
# "Any periodic function can be decomposed into a sum of sine and cosine waves."
# 
# This is a very simple statement but its implications are very significant. For example, shown below are the functions sin(2x) and cos(3x) and their corresponding summation:

# %%
import plotly.graph_objs as go
import numpy as np

x = np.linspace(0, 3 * np.pi, 500)

y1 = np.sin(2 * x)
y2 = np.cos(3 * x)
y_sum = y1 + y2

trace1 = go.Scatter(x=x, y=y1, mode='lines',name='sine(2x)', line=dict(color='blue'))
trace2 = go.Scatter(x=x, y=y2, mode='lines', name='cos(3x)', line=dict(color='green'))
trace3 = go.Scatter(x=x, y=y_sum, mode='lines', name='sum', line=dict(color='red'))

layout = go.Layout(
    title='Example Sum of Sinusoidal Waves',
    xaxis=dict(title='X'),
    yaxis=dict(title='Y')
)

data = [trace1, trace2, trace3]

fig = go.Figure(data=data, layout=layout)

fig.show()


# %% [markdown]
# Notice that the functions of sin(2x) and cos(3x) are very uniform and simple functions yet their summation (red line) leads to a more complex pattern. This is the main idea behind the Fourier series.
# 
# We can even use the Fourier series to construct a square wave by summing sine waves (harmonics) of different odd number frequencies and amplitudes:

# %% [markdown]
# ![Screenshot%202023-11-26%20at%2013.36.40.png](attachment:Screenshot%202023-11-26%20at%2013.36.40.png)

# %%
import plotly.graph_objs as go
import numpy as np

x = np.linspace(0, 3 * np.pi, 1000)
y = np.array([np.sin((2*k + 1) * x) / (2*k + 1) for k in range(100)]).sum(axis=0) * (4 / np.pi)

trace = go.Scatter(x=x, y=y, mode='lines', name='Square Wave', line=dict(color='blue'))

layout = go.Layout(
    title='Square Wave',
    xaxis=dict(title='X'),
    yaxis=dict(title='Y', range=[-1.5, 1.5])
)

data = [trace]

fig = go.Figure(data=data, layout=layout)

fig.show()

# %% [markdown]
# What’s staggering about this result is that we have generated a sharp and straight line plot from smooth sine functions. This shows the true power of the Fourier series to construct any periodic function.

# %% [markdown]
# ### Theory

# %% [markdown]
# As we said above, the Fourier series states that any periodic function can be broken down into a sum of sine and cosine waves. Mathematically, this is written as:

# %% [markdown]
# ![Screenshot%202023-11-26%20at%2013.39.07.png](attachment:Screenshot%202023-11-26%20at%2013.39.07.png)

# %% [markdown]
# Where:
# - A_0: average value of the given periodic function
# - A_n: coefficients of the cosine components
# - B_n: coefficients of the sine components
# - n: the order which is the frequency of the sine or cosine wave, this is referred to as the ‘harmonics’
# - P: period of the function

# %% [markdown]
# The period, P, and order, n, are known ahead of time. However, the coefficients (A_0, A_n, B_n) need to be calculated to determine which sine and cosine components combined produce the given periodic function. Luckily most Python data science packages do this process for us!

# %% [markdown]
# ### Link to Forecasting
# 

# %% [markdown]
# Are you wondering how does the Fourier series fit into time series forecasting? Well, remember that Fourier series deal with periodic functions and we often find that time series contain some periodic structure (typically seasonality). Therefore, we can use the Fourier series to model any complex seasonal pattern in our time series data!
# 
# Pros of using the Fourier series to model seasonality are:
# - Any season length
# - Model multiple seasonal patterns
# - The sensitivity of the Fourier seasonality can be tuned through the order and amplitudes of the sine and cosine components
# - Computationally efficient when seasonal periods are greater than ~200
# 
# Many of these advantges cannot be achieved with the SARIMA model as it only accepts integer seasonality, a single season, and often runs out of memory when the seasonal period is more than ~200.
# 
# Cons of using the Fourier series to model seasonality are:
# - Assumes seasonal patterns and cycles remain fixed
# 

# %% [markdown]
# ## ARIMAX & Exogenous Features
# 

# %% [markdown]
# ### Intuition

# %% [markdown]
# For ARIMA models we can add extra external features to aid in the forecasting. These features are called exogenous features and make the ARIMA model become an ARIMAX model. For example, we may use the current interest rates as an exogenous feature when forecasting the value of a house.
# 
# You can think of the ARIMAX model as just like regular linear regression with the addition of autoregressors and moving-average components (endogenous variables). The trick here is to allow the Fourier series to be one of these exogenous features or an explanatory variable as is often described in linear regression.
# 
# As we are dealing with time series, the exogenous features need to be time indexed just like the autoregressors and moving-averages. They also need to be known at the point of the forecast. For example, if we want to forecast the value of a house in May, we need to know what the interest rates are in May if we want them as an exogenous feature.

# %% [markdown]
# ### Theory

# %% [markdown]
# Mathematically, the exogenous features are added to the classic ARIMA model in the following way:

# %% [markdown]
# ![Screenshot%202023-11-26%20at%2013.51.04.png](attachment:Screenshot%202023-11-26%20at%2013.51.04.png)

# %% [markdown]
# - y: time-series/lags at different time steps
# - x: exogenous feature
# - β: coefficient for exogenous feature
# - ϕ: coefficients of the autoregressive components (lags)
# - p: number of autoregressive components
# - ε: forecast error terms, the moving-average components
# - θ: coefficients of the lagged forecast errors
# - q: number of lagged error components

# %% [markdown]
# ### Fourier Series Features
# 

# %% [markdown]
# To add the Fourier series as exogenous to an ARIMA model is relatively simple as the coefficients/amplitudes, β, are deduced for us and all we need to provide are the corresponding sine and cosine terms. In pseudo-code, this is equivalent to:

# %%
sin(2*pi*frequency*time_index/period)

cos(2*pi*frequency*time_index/period)

# %% [markdown]
# As an example, let’s say we have monthly data with a yearly seasonality and we want the Fourier components for May. This, in pseudo-code, would be:

# %%
sin(2*pi*frequency*5/12)

cos(2*pi*frequency*5/12)

# %% [markdown]
# However, we still have the frequency (the order) value to deduce. This is typically found by passing numerous sine and cosine component orders and letting the model find the most useful ones.

# %% [markdown]
# ### Python Walkthrough
# 
# 

# %%
# Import packages
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pmdarima as pm
import numpy as np

# Read in the data
data = pd.read_csv('AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
data['month_num'] = data['Month'].dt.month

# Stabilise the variance
data['Passengers_Boxcox'], lam = boxcox(data['#Passengers'])
data.dropna(inplace=True)

# Get fourier features
for order in range(1, 10):
    data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * data['month_num'] / 12)
    data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * data['month_num'] / 12)

# name of fourier features
fourier_features = [i for i in list(data) if i.startswith('fourier')]

# Split train and test
train = data.iloc[:-int(len(data) * 0.2)]
test = data.iloc[-int(len(data) * 0.2):]

# Build auto-ARIMA model with fourier features
model = pm.auto_arima(train['Passengers_Boxcox'],
                      X=train[fourier_features],
                      seasonal=False,
                      stepwise=True,
                      suppress_warnings=True,
                      max_order=None,
                      information_criterion='aicc',
                      error_action="ignore")

# Get the forecasts and apply inverse box-cox transform
boxcox_forecasts = model.predict(n_periods=len(test), X=test[fourier_features])
forecasts = inv_boxcox(boxcox_forecasts, lam)


def plot_forecasts(forecasts: list[float], title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Month'], y=train['#Passengers'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Month'], y=test['#Passengers'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Month'], y=forecasts, name='Forecast'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=650, title_x=0.5, height=400, xaxis_title='Date',
                      yaxis_title='Passenger Volume')

    return fig.show()


# Plot the forecasts
plot_forecasts(forecasts, 'Harmonic Regression')


# %% [markdown]
# # Summary

# %% [markdown]
# When the seasonality of your time series is a non-integer, has numerous patterns, or is very long (>50 points) then it is preferable to use the Fourier series to model this seasonality component. This can be achieved by adding the Fourier series as an exogenous feature to a regular ARIMA model to make it an ARIMAX. These exogenous features are external covariates that aid in the forecasting of the time series.


