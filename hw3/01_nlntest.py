# %% [markdown]
# # آزمون خطی بودن

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nlntest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import bds
import yfinance as yf       # https://pypi.org/project/yfinance/
import pytse_client as tse # https://pypi.org/project/pytse-client/

# %% [markdown]
# ## دریافت داده
# 
# این بخش در تمرین شماره یک انجام گرفته و از همان کدها استفاده می‌شود.

# %% [markdown]
# دیتای اینتل و والمارت از api یاهوفایننس دریافت می‌شود:

# %%
tickers = ['INTC', 'WMT']
data_nse = yf.download(tickers, group_by = 'ticker', start="2017-01-01", end="2022-11-19")

# %% [markdown]
# دیتای دو سهم ایرانی شپنا و وبصادر نیز از سایت tse اسکریپ می‌شود:

# %%
tickers = ['خودرو', 'شپنا']
prices_dict = tse.download(symbols=tickers, adjust=True)
prices_dict_reform = {(outerKey, innerKey):
                             values for outerKey, innerDict 
                             in prices_dict.items() for innerKey, values 
                             in innerDict.iteritems()}
data_tse = pd.DataFrame(prices_dict_reform)
d = {'خودرو': 'Khodro', 'شپنا': 'Shepna'}
data_tse = data_tse.rename(columns=d, level=0)

# %% [markdown]
# ## آشنایی با داده

# %% [markdown]
# نگاهی به دیتای شرکت اینتل می‌اندازیم:

# %%
data_nse['INTC'][['Open', 'High', 'Low', 'Close', 'Adj Close']].tail()

# %% [markdown]
# نگاهی به دیتای شرکت خودرو می‌اندازیم:

# %%
cols = ['open', 'high', 'low', 'close', 'adjClose']
data_tse['Khodro'].set_index('date')[cols].tail()

# %% [markdown]
# نمودار شرکت اینتل را در یک سال آخر رسم می‌کنیم تا با حرکت کلی سهم آشنا شویم:

# %%
data_nse['INTC', 'Adj Close'][-300:].plot(title='Adj Close for INTC', figsize=(12,5))

# %% [markdown]
# ## مدل‌سازی
# 
# ### تست ایستا بودن دیفرنس قیمت‌ها

# %% [markdown]
# برای اینکه بدانیم آیا سری قیمت‌ها با یک‌بار دیفرنس گرفتن ایستا می‌شوند، تست ADF را انجام می‌دهیم و در صورتی که pvalue این تست کم‌تر از پنج‌درصد باشد، فرض صفر را رد می‌کنیم و سری را ایستا در نظر می‌گیریم.
# 
# در صورتی که سری با یک بار دیفرنس گرفتن ایستا شود، 
# بعدا در مدل ARIMA مقدار d را برابر با یک قرار می‌دهیم.

# %% [markdown]
# :تست ایستا بودن دیفرنس قیمت‌های اینتل

# %%
diff_adjClose = data_nse['INTC', 'Adj Close'].diff()
adfuller(np.array(diff_adjClose)[1:])[1] < 0.05

# %% [markdown]
# :تست ایستا بودن دیفرنس قیمت‌های والمارت

# %%
diff_adjClose = data_nse['WMT', 'Adj Close'].diff()
adfuller(np.array(diff_adjClose)[1:])[1] < 0.05

# %% [markdown]
# :تست ایستا بودن دیفرنس قیمت‌های خودرو

# %%
diff_adjClose = data_tse['Khodro', 'adjClose'].diff()
diff_adjClose = diff_adjClose[~np.isnan(diff_adjClose)]
adfuller(diff_adjClose) [1] < 0.05

# %% [markdown]
# :تست ایستا بودن دیفرنس قیمت‌های شپنا

# %%
diff_adjClose = data_tse['Shepna', 'adjClose'].diff()
diff_adjClose = diff_adjClose[~np.isnan(diff_adjClose)]
adfuller(diff_adjClose) [1] < 0.05

# %% [markdown]
# همگی سری‌های قیمتی با یک بار دیفرنس گرفتن ایستا می‌شوند؛ بنابراین می‌توانیم در مدل ARIMA مقدار d را برای همگی یک در نظر بگیریم.
# 
# 

# %% [markdown]
# برای یافتن بهترین مدل، یک تابع می‌نویسیم. این تابع مدل‌های ARIMA مختلف را امتحان می‌کند و مدلی را که کم‌ترین aic دارد به عنوان خروجی پس می‌دهد.

# %%
def best_arima(data):
    p_max = 12
    min_aic = np.inf
    best_model = None
    for p in range(1, p_max):
        model = ARIMA(data, order=(p, 1, 0)).fit()
        if min_aic > model.aic:
            min_aic = model.aic
            best_model = model
        
    return best_model

# %% [markdown]
# ## نتایج

# %% [markdown]
# ### سهم خودرو
# 
# پس از گرفتن مانده‌ بهترین مدل ARIMA تست خطی بودن را اجرا می‌کنیم.
# 
# نتیجه: همه آزمون‌های خطی، فرض صفر را رد کرده‌اند. بنابراین سری قیمتی خودرو، رفتاری غیرخطی دارد.

# %%
symbol = 'Khodro'
not_null_data = data_tse[data_tse[symbol, 'adjClose'].notna()][symbol, 'adjClose']
arim_model = best_arima(not_null_data)
residuals1 = arim_model.resid
nlntest.nlntstuniv(np.array(residuals1))

# %% [markdown]
# ### سهم شپنا
# 
# پس از گرفتن مانده‌ بهترین مدل ARIMA تست خطی بودن را اجرا می‌کنیم.
# 
# نتیجه: همه آزمون‌های خطی، فرض صفر را رد کرده‌اند. بنابراین سری قیمتی شپنا، رفتاری غیرخطی دارد.

# %%
symbol = 'Shepna'
not_null_data = data_tse[data_tse[symbol, 'adjClose'].notna()][symbol, 'adjClose']
arim_model = best_arima(not_null_data)
residuals = arim_model.resid
nlntest.nlntstuniv(np.array(residuals))

# %% [markdown]
# ### سهم اینتل
# 
# پس از گرفتن مانده‌ بهترین مدل ARIMA تست خطی بودن را اجرا می‌کنیم.
# 
# نتیجه: هیچ‌کدام از آزمون‌های خطی، فرض صفر را رد نکرده‌اند. بنابراین سری قیمتی اینتل، رفتاری خطی دارد.

# %%
symbol = 'INTC'
not_null_data = data_nse[data_nse[symbol, 'Adj Close'].notna()][symbol, 'Adj Close']
not_null_data = np.array(not_null_data)
arim_model = best_arima(not_null_data)
residuals = arim_model.resid
nlntest.nlntstuniv(np.array(residuals))

# %% [markdown]
# ### سهم والمارت
# 
# پس از گرفتن مانده‌ بهترین مدل ARIMA تست خطی بودن را اجرا می‌کنیم.
# 
# نتیجه: آزمون ترسورتا فرض صفر را کرده است. اما سه آزمون دیگر در سطح اطمینان ۹۵ درصد فرض صفر را رد نکرده‌اند. هر چند تست رمزی نیز بسیار به مقدار بحرانی نیزدیک است.
# در مورد سری قیمتی والمارت نمی‌توان با قطعیت نظر دارد.

# %%
symbol = 'WMT'
not_null_data = data_nse[data_nse[symbol, 'Adj Close'].notna()][symbol, 'Adj Close']
not_null_data = np.array(not_null_data)
arim_model = best_arima(not_null_data)
residuals = arim_model.resid
nlntest.nlntstuniv(np.array(residuals))


