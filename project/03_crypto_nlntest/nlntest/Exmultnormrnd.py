# Running package on multivariate normal random number
import numpy as np
import pandas as pd
import nlntest as nlt


# Example: multivariate normal random numbers
y3=np.random.randn(1000,3)
print(' ')
print('Example: Linearity test of multivariate time series generated by the normal distribution')
print(' ')
resultsMulti=nlt.nlntstmultv(y3)
resultsAnn=nlt.annnlntst(y3)

''' Since the random series are not dependent, pvalues will greater than 0.05
in 95% of times that you run the modules '''


