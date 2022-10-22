# %% [markdown]
# # دریافت اطلاعات نمادهای بورس نیویورک به کمک یاهو-فایننس

# %%
import pandas as pd
import yfinance as yf       # https://pypi.org/project/yfinance/

# %% [markdown]
# ## دریافت تاریخچه قیمتی نمادها

# %% [markdown]
# اطلاعات مربوط به این هشت سهم را یک جا دریافت می‌کنیم:
# 
# اپل، برک‌شایر-هث‌وی، گوگل، اینتل، کوکاکولا، مایکروسافت، تی‌اند‌تی، والمارت

# %%
tickers = ['AAPL', 'BRK', 'GOOG', 'INTC', 'KO', 'MSFT', 'T', 'WMT']
data = yf.download(tickers, group_by = 'ticker', start="2017-01-01", end="2022-10-17")

# %% [markdown]
# متغیر data یک دیتافریم multi-index است که می‌توانیم اطلاعات هر کدام از هشت سهم را جدا کنیم و ببینیم:

# %%
data['MSFT'][['Open', 'High', 'Low', 'Close', 'Adj Close']].tail()

# %%
data['MSFT', 'Adj Close'][-300:].plot(title='Adj Close for MSFT', figsize=(12,5))

# %% [markdown]
# ### نوشتن خروجی در فایل اکسل در شیت‌های جداگانه

# %% [markdown]
# برای نوشتن در خروجی، ابتدا اسم فایل و فولدر مربوطه را انتخاب می‌کنیم. سپس در صورت موجود نبودن فولدر، آن را می‌سازیم.
# 
# خروجی این مرحله در این مقصد ذخیره می‌شود:
# ./excel_files/01_NYSE_prices.xlsx

# %%
import os

dir_name = 'excel_files'
os.makedirs(rf"./{dir_name}", exist_ok=True)
output_file_name = '01_NYSE_prices'
path = rf"./{dir_name}/{output_file_name}.xlsx"

# %% [markdown]
# در یک حلقه for هر کدام از سهم‌ها را در شیت جداگانه‌ای می‌نویسیم.

# %%
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
for col in data.columns.levels[0].tolist():
    data[col].to_excel(writer, sheet_name = col)
writer.save()

# %% [markdown]
# ## دریافت اطلاعات ترازنامه‌ نمادها

# %% [markdown]
# ابتدا یک تابع برای نشان دادن پروگرس دریافت ترازنامه‌ها تعریف می‌کنیم.

# %%
def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '=' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

# %% [markdown]
# در این مرحله، تک به تک ترازنامه هر کدام از شرکت‌ها دریافت شده و در یک شیت جداگانه نوشته می‌شود. در صورت بروز خطا برای یک نماد، حلقه متوقف نمی‌شود و فقط پیامی در خروجی چاپ می‌شود.
# 
# خروجی این مرحله در این مقصد ذخیره می‌شود:
# ./excel_files/01_NYSE_balance_sheets.xlsx

# %%
output_file_name = '01_NYSE_balance_sheets'
path = rf"./{dir_name}/{output_file_name}.xlsx"
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')

progressBar(0, len(tickers))
for i, ticker in enumerate(tickers):
    try:
        df_balance_sheet = yf.Ticker(ticker).balance_sheet
        df_balance_sheet.to_excel(writer, sheet_name=ticker)
        progressBar(i+1, len(tickers))
    except:
        print(f"failed to fetch the balance_sheet of '{ticker}'")
writer.save()

# %% [markdown]
# نمونه ترازنامه دریافت شده (ترازنامه والمارت):

# %%
df_balance_sheet.head()


