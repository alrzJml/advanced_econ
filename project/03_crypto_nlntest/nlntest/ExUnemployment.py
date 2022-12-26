# Running the package on US monthly unemployment rate
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import nlntest as nlt



# Example: Unemployment Rate, Percent, Monthly,
# Seasonally Adjusted data in an excel file format for 1948:01-2022:08
# Data Source: https://fred.stlouisfed.org/series/UNRATE


print(' ')
print('         '      'Example: Linearity Test of Unemployment Rate of USA')
print(' ')
df=pdr.get_data_fred('UNRATE','1948-01-01','2022-08-01')
print(df)
y=df.values
results=nlt.nlntstuniv(y)
resultsMulti=nlt.nlntstmultv(y)
resultsAnn=nlt.annnlntst(y)

''' Since the unemployment rate is a nonlinear process pvalues will be smaller 
than 0.05 in 95% of times that you run the modules'''



# Note: For running this example you need the internet connection.






