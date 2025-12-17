import pandas as pd

import yfinance as yf

def load_sp500_data(start='2010-01-01', end='2025-01-01'):

    data = yf.download('^GSPC', start=start, end=end, auto_adjust=True)
    
    data = data.reset_index()

    data['Date'] = pd.to_datetime(data['Date'])

    data = data.sort_values('Date').reset_index(drop=True)


    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)
    
    return data