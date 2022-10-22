# %% [markdown]
# # مدل ترش‌هولد برای سهام مایکروسافت

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot

from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from bds import bds

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ابتدا فایل اطلاعات قیمتی را که در تمرین اول دریافت کردیم، می‌خوانیم: 

# %%
df_msft = pd.read_excel('./excel_files/01_NYSE_prices.xlsx', sheet_name='MSFT')

# %%
df_msft[['Date', 'Adj Close']]

# %% [markdown]
# ## آماده‌سازی داده

# %% [markdown]
# سری زمانی قیمت‌های تعدیل شده را می‌سازیم:

# %%
prices_series = df_msft.set_index('Date')['Adj Close']

# %% [markdown]
# برای گپ‌های موجود به ترتیب این کارها را می‌کنیم:
# 
# 1. فرکانس سری زمانی را روزانه می‌کنیم.
#   
# 2. با متد ffill مقادیر نال به وجود آمده را پر می‌کنیم.
#   
# 3. چون برای همه شنبه‌ها و یکشنبه‌ها، دیتایی در دسترس نبوده، بنابراین این دو روز را از سری زمانی حذف می‌کنیم.

# %%
prices_series = prices_series['2018':].asfreq('1D').ffill()
prices_series = prices_series[prices_series.index.weekday<5]

# %% [markdown]
# نگاهی به سری زمانی می‌اندازیم. این سری ناماناست.

# %%
prices_series.plot(title='MSFT Adj Close', figsize=(8, 4))

# %% [markdown]
# از سری دیفرنس می‌گیریم و نمودار آن را رسم می‌کنیم. به نظر می‌رسد با یک بار دیفرنس گرفتن، سری مانا شده است.

# %%
diff_prices = prices_series.diff()
diff_prices = diff_prices['2018':]
diff_prices.plot(title='MSFT Diff Adj Close', figsize=(8, 4))

# %% [markdown]
# تست ADF را برای مانایی اجرا می‌کنیم. خروجی دوم، مقدار p-value را برای این تست نمایش نمی‌دهد. سری ماناست..

# %%
adfuller(diff_prices.reset_index()['Adj Close'].dropna())

# %% [markdown]
# بنابراین مقدار d در مدل ARIMA برابر با یک است.
# 
# حال مدل $ARIMA(p, 1, 0)$ را برای سری زمانی به کار می‌گیریم.

# %% [markdown]
# ## مدل‌ AR
# اگر بتونیم سری را با مدل AR مدل کنیم و مانده‌های مدل نیز i.i.d باشند، می‌توان نتیجه گرفت که سری رفتار غیرخطی ندارد.
# 
# بنابراین ابتدا باید مطمئن شویم که مانده‌های مدل AR یک سری i.i.d نیست.

# %% [markdown]
# تابع pacf سری را نمایش می‌دهیم:

# %%
plot_pacf(diff_prices.reset_index()['Adj Close'].dropna())
pyplot.show()

# %% [markdown]
# برای پیدا کردن بهترین مرتبه مدل AR این تابع را می‌نویسیم:

# %%
def auto_ar_model(values, max_p=12):
    best_orders = None
    _best_aic = np.Inf
    for p in range(1, max_p+1):
        model = ARIMA(values, order=(p,1,0))
        results = model.fit()
        if results.aic < _best_aic:
            _best_aic = results.aic
            best_orders = model.order
    return best_orders

# %% [markdown]
# به کمک تابع بالا، بهترین مرتبه مدل AR را به دست می‌اوریم. به نظر می‌رسید مدل AR(10) بهترین مدل از نظر معیار AIC است.

# %%
best_order = auto_ar_model(prices_series)
best_order

# %% [markdown]
# مدل‌سازی را به کمک بهترین مرتبه انجام می‌دهیم:

# %%
model = ARIMA(prices_series, order=best_order)
results = model.fit()

# %% [markdown]
# نتایج مدل AR(10) در زیر به طور خلاصه آورده شده است.
# 
# ضرایب لگ‌های اول، ششم، هشتم و دهم معنادارند.

# %%
results.summary()

# %% [markdown]
# نمودار مانده را می‌کشیم:

# %%
results.resid.plot()

# %% [markdown]
# تست BDS را برای i.i.d بودن مانده اجرا می‌کنیم. فرض صفر این تست، i.i.d بودن فرایند است. مقادیر p-value بسیار کوچک‌اند و بنابراین می‌توانیم فرض صفر را رد کنیم. این موضوع موجب می‌شود که بتوانیم امکان روابط غیرخطی در فرایند را بررسی کنیم:

# %%
bds(results.resid, 3)

# %% [markdown]
# ## مدل TAR

# %% [markdown]
# مدل TAR در هیچ پکیج پایتونی پیاده‌سازی نشده است.
# بنابراین باید این کار را شخصا انجام دهیم.
# 
# تابع زیر یک سری زمانی را به همراه لیستی از ترش‌هولدها به عنوان ورودی دریافت می‌کند و برای همه مقادیر ممکن، مدل‌سازی را انجام می‌دهد. سپس MSE هر مدل را در فایل
# 03_TAR_MSE_LOG.txt
# می‌نویسد تا بعدا مورد استفاده قرار بگیرند:
# 
# این تابع به این طریق کار می‌کند:
# 
# 1. ابتدا مقادیر AR(p) را برای به ازای‌ مرتبه‌های مختلف برای سری زمانی ذخیره می‌کند.
# 
# 2. سپس به ازای همه AR های ممکن در همه ترش‌هولدها خطای MSE را حساب می‌کند و در فایل لاگ ذخیره می‌کند.
# 
# برای مثال هنگامی که بیشینه مرتبه مدل AR را پنج انتخاب می‌کنیم و می‌خواهیم مدل‌مان دو ترش‌هولد (سه بخشی) داشته باشد، این تابع مقدار پنج به توان سه یا همان ۱۲۵ خطا را محاسبه می‌کند و در فایل لاگ می‌نویسد.

# %%
import itertools

def switching_treshold_model(data, tresholds, max_p=5):
    data = pd.DataFrame(data)
    col_name = data.columns[0]

    # caluclate AR(p) model
    for p in range(1, max_p+1):
        model = ARIMA(data[col_name], order=(p, 1, 0))
        results = model.fit()
        data[f'ar_{p}'] = results.predict()

    
    iterate_matrix = []
    for d in range(len(tresholds)+1):
        iterate_matrix = iterate_matrix.__add__([range(1, max_p+1)])

    f = open('03_TAR_MSE_LOG.txt', 'a')

    for orders in itertools.product(*iterate_matrix):
        switching_pred = data.loc[data[col_name].diff().shift(1) <= tresholds[0], f'ar_{orders[0]}']
        for i in range(len(tresholds)):
            lower_band = tresholds[i]
            upper_band = tresholds[i+1] if len(tresholds)>i+1 else np.Inf
            ar_tmp = data.loc[(data[col_name].diff().shift(1) > lower_band) & (data[col_name].diff().shift(1) <= upper_band), f'ar_{orders[i+1]}']
            switching_pred = switching_pred.append(ar_tmp).sort_index()
        
        err = mean_squared_error(data[col_name][2:], switching_pred)
        f.write(f'{orders} = {err}\n')
    f.close()

# %% [markdown]
# تابع بالا را برای دو ترش‌هولد ران می‌کنیم:
# 
# 1. ترش‌هولد برابر با صفر: زمانی که در روز معاملاتی قبل، بازده مثبت یا منفی بوده
# 
# 2. ترش‌هولد دوتایی: زمانی که بازده نزدیک صفر بوده یا با آن فاصله مثبت/منفی داشته است.

# %%
switching_treshold_model(prices_series, tresholds=[0])
switching_treshold_model(prices_series, tresholds=[-1.8, 1.8])

# %% [markdown]
# لاگ را می‌خوانیم و مدلی که کم‌ترین MSE داشته را به عنوان مدل نهایی انتخاب می‌کنیم:

# %%
with open('03_TAR_MSE_LOG.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

tar_orders_mse = [[x.split(' = ')[0], x.split(' = ')[1]] for x in lines]
least_mse = np.Inf
for i, elem in enumerate(tar_orders_mse):
    if float(elem[-1]) < least_mse:
        best_mse_index = i
        least_mse = float(elem[-1])

# %%
tar_orders_mse[best_mse_index][0]

# %% [markdown]
# با اینکه مدل ۲ ترش‌هولد دارد (سه بخشی است) اما بخش دوم و سوم یک مدل مشترک را نمایش می‌دهند: AR(5)

# %% [markdown]
# بنابراین مدل نهایی دو بخشی خواهد بود. مقدار ترش‌هولد $-1.8$ و به ترتیب مدل‌های AR(3) و AR(5) برای این دو بخش مناسب خواهند بود.

# %%
model3 = ARIMA(prices_series, order=(3, 1, 0))
results3 = model3.fit()

model5 = ARIMA(prices_series, order=(5, 1, 0))
results5 = model5.fit()

price_tar_df = pd.DataFrame(prices_series)
price_tar_df['ar_5_pred'] = results5.predict()
price_tar_df['ar_3_pred'] = results3.predict()

price_tar_df.loc[price_tar_df['Adj Close'].shift(1) <= -1.8, f'switching_pred'] = price_tar_df['ar_3_pred']
price_tar_df.loc[price_tar_df['Adj Close'].shift(1) > -1.8, f'switching_pred'] = price_tar_df['ar_5_pred']

# %% [markdown]
# نمودار قیمتی تعادلی را به همراه مدل ترش‌هولد می‌کشیم. به نظر می‌رسد که مدل ترش‌هولد مقادیر قبلی سری زمانی را با کمی تغییر کپی می‌کند.

# %%
price_tar_df[-50:].plot(figsize=(12, 5))

# %% [markdown]
# اما با با محسابه MSE مدل پیش‌بینی naive و مدل TAR می‌بینیم که خطای مدل TAR کم‌تر است.

# %%
naive_err = mean_squared_error(prices_series.shift(1).dropna(), prices_series[1:])
tar_err = tar_orders_mse[best_mse_index][-1]
print(f'naive_err: {naive_err} \ntar_err: {tar_err}')

# %% [markdown]
# دیتافریم پیش‌بینی نهایی:

# %%
price_tar_df


