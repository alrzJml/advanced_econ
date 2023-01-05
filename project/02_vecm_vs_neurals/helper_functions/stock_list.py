import requests
import pandas as pd

from pytickersymbols import PyTickerSymbols


def _get_tehran50_stock_list():
    """Return the list of symbols in Tehran 50 index."""

    url = "https://www.fipiran.com/IndexDetails/_IndexInstrument?Lval30=69932667409721265"
    x = requests.get(url)
    stock_list = pd.read_html(x.text)[0]['نماد'].tolist()
    return stock_list


def get_stock_list(index: str = 'Teh50'):
    """Return the list of symbols in the given index."""

    if index == 'Teh50':
        return _get_tehran50_stock_list()

    stock_data = PyTickerSymbols()
    stock_list = list(stock_data.get_stocks_by_index(index))

    stocks_in_index = []
    for stock in stock_list:
        for symbol in stock['symbols']:
            if symbol['currency'] == 'USD':
                stocks_in_index.append(symbol['yahoo'])
                break
        else:
            continue

    return stocks_in_index
