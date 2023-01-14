import pandas as pd
import yfinance as yf       
import pytse_client as tse
import numpy as np

def _get_tse_prices(tickers):
    prices_dict = tse.download(symbols=tickers, adjust=True)
    prices_dict_reform = {(outerKey, innerKey):
                                values for outerKey, innerDict 
                                in prices_dict.items() for innerKey, values 
                                in innerDict.iteritems()}
    df_raw = pd.DataFrame(prices_dict_reform)

    dates_df = df_raw.loc[:, df_raw.columns.get_level_values(1)=='date']
    min_date, max_date = np.min(np.min(dates_df)), np.max(np.max(dates_df))

    df = pd.DataFrame()
    df['Date'] = pd.date_range(min_date, max_date)

    for symbol in df_raw.columns.levels[0]:
        df_tmp = df_raw[symbol]
        df_tmp = df_tmp[['date', 'adjClose']]
        df_tmp.columns = ['Date', symbol]
        df = df.merge(df_tmp, on='Date', how='left')

    df = df.set_index('Date')

    return df


def get_prices(tickers, is_tse=False):
    
    if is_tse:
        return _get_tse_prices(tickers)
    else:    
        df_raw = yf.download(tickers, group_by = 'ticker', period = '100y', interval='1d')
        df = df_raw.loc[:, df_raw.columns.get_level_values(1)=='Adj Close']
        df.columns = [x[0] for x in df.columns.tolist()]
        return df