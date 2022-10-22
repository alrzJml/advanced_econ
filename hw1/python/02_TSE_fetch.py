# %% [markdown]
# # دریافت اطلاعات نمادهای بورس تهران 

# %%
import pandas as pd     
import pytse_client as tse # https://pypi.org/project/pytse-client/

# %% [markdown]
# ## دریافت قیمت نمادها

# %% [markdown]
# قیمت‌های ابتدایی، پایانی، بیشترین و کم‌ترین به همراه تعدیل‌شده‌شان را برای ۶ سهم دریافت می‌کنیم.
# شش سهم: ***شپنا، کگل، خاور، کرمان، وبصادر و فارس***
# 
# برای اینکه کار کردن با دیتای خروجی راحت‌تر شود، سعی می‌کنیم تا خروجی را به یک multi-index دیتافریم تبدیل کنیم. (همانند خروجی پکیج yfinance هنگامی که دیتای چند نماد را با هم دریافت می‌کنیم) 

# %%
tickers = ['شپنا', 'کگل', 'خاور', 'کرمان', 'وبصادر', 'فارس']
prices_dict = tse.download(symbols=tickers, adjust=True)
prices_dict_reform = {(outerKey, innerKey):
                             values for outerKey, innerDict 
                             in prices_dict.items() for innerKey, values 
                             in innerDict.iteritems()}
data = pd.DataFrame(prices_dict_reform)

# %% [markdown]
# دیتای همه ۶ نماد در یک دیتافریم ذخیره شده‌اند و می‌توانیم به این صورت، مثلا دیتای سهم «**خاور**» را ببینیم:

# %%
cols = ['open', 'high', 'low', 'close', 'adjClose']
data['خاور'].set_index('date')[cols].tail()

# %% [markdown]
# برای نوشتن در خروجی، ابتدا اسم فایل و فولدر مربوطه را انتخاب می‌کنیم. سپس در صورت موجود نبودن فولدر، آن را می‌سازیم.
# 
# خروجی این مرحله در این مقصد ذخیره می‌شود:
# ./excel_files/02_TSE_prices.xlsx

# %%
import os

dir_name = 'excel_files'
os.makedirs(rf"./{dir_name}", exist_ok=True)
output_file_name = '02_TSE_prices'
path = rf"./{dir_name}/{output_file_name}.xlsx"

# %% [markdown]
# در یک حلقه for هر کدام از سهم‌ها را در شیت جداگانه‌ای می‌نویسیم.

# %%
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
for col in data.columns.levels[0].tolist():
    data[col].to_excel(writer, sheet_name = col)
writer.save()


