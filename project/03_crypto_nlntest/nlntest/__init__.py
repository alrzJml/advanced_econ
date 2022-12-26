
''' Description:
 nonlinearity test for univariate and multivariate time series. Matlab version of
 this module has been used in S. Mohammadi(2019).The test is a generalization of Lee,
 White and Granger (1993) test. In addition to generalization to multivariate
 time series, some modifications are made in Mohammadi(2019) for increasing
 the power of Lee, White and Granger (1993)test in the case of a univariate
 time series.
 
 
Input:
 y: a univariate(multivariate) time series in the form column
 vector(s)
 
Output: 
 Probability value of the test. Reject linearity(H0) if pval is
 less than 5%(or any predetermined level of significance).


 References:

 1-  Mohammadi S. Neural network for univariate and multivariate nonlinearity tests. 
     Stat Anal Data Min: The ASA DataSci Journal. 2019.
     13:50-70.https://doi.org/10.1002/sam.11441

 Copyright, Shapour Mohammadi 2022.shmohmad@ut.ac.ir'''
 
#--------------------------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels as stats
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn import decomposition
import math
from sklearn.neural_network import MLPRegressor
import math
import sklearn
from sklearn import preprocessing

def nlnTstmols(y,x):
 dimy=np.shape(y)
 if len(dimy)==1:
  cy=1
 else:
  cy=dimy[1]  
 dimx=np.shape(x)
 if len(dimx)==1:
     cx=1
 else:
  cx=dimx[1]
 rx=dimx[0]        
 x=np.column_stack((np.ones((rx,1)), x))
 xT=np.transpose(x)
 betaols=np.zeros((cx+1,cy))
 yfit=np.zeros((rx,cy))
 resmdl=np.zeros((rx,cy))
 for i in range(0,cy):
   betaols[:,i]=np.linalg.inv(xT@x)@(xT@y[:,i])
   yfit[:,i]=x@betaols[:,i]
   resmdl[:,i]=y[:,i]-yfit[:,i]
 return resmdl,yfit,betaols

def annnlntst(y):

 scale=sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
 scaler=scale.fit(y)
 y=scaler.transform(y)
 
# Finding optimal lag.
 T=len(y)
 aiccriter=np.zeros((5,1))
 for lag in range(1,6):
    xlagmat=lagmat(y,maxlag=lag,trim='None',original='ex')
    xlagmat=xlagmat[lag:T,:]
    yadj=y[lag:T,:]
    (resmdl,yfit,betaols)=nlnTstmols(yadj,xlagmat)
    aiccriter[lag-1,0]=np.log(np.linalg.det((np.transpose(resmdl)@resmdl))/(T-lag))+2*lag/(T-lag)

 optlag=aiccriter.argmin()+1
 X=lagmat(y,maxlag=optlag,trim='None',original='ex')
 X=X[optlag:T,:]
 yadj=y[optlag:T,:]

 dimX=np.shape(X)
 dimyadj=np.shape(yadj)
 if len(dimX)==1:
  X=X.reshape(-1,1)
 if len(dimyadj)==1:
    Ky=1
 else:
    Ky=dimyadj[1]   
 if len(dimX)==1:
   K=1
 else:
  K=dimX[1]
 T=len(X)
#Estimate a linear model and get residuals.
 (resid,yfit,beta)=nlnTstmols(yadj,X)
 dimresid=np.shape(resid)

 if len(dimresid)==1:
    cresid=1
    resid2=np.ravel(resid)
 else:
     if dimresid[1]==1:
         resid2=np.ravel(resid)
         cresid=dimresid[1]
     else:
      cresid=dimresid[1]
      resid2=resid
    
# Number of hidden units(neurons).
 nernum=10

# Preallocation for neural networks weights.
 netnum=100
 bias1l=[]
 bias2l=[]
 bias1u=[]
 bias2u=[]
 IWtotal1l=[]
 IWtotal1u=[]
 LWtotal2l=[] 
 LWtotal2u=[]

   
# Training 100 neural networks for obtaining support space of neural 
# network weights.

 for i in range(1,netnum+1):
    mdlann=MLPRegressor(hidden_layer_sizes=(nernum,nernum),
                    activation='tanh',max_iter=10000)
    net=mdlann.fit(X,resid2)
    netW=net.coefs_
    Bias=net.intercepts_
    B0=np.reshape(Bias[0],(nernum,1))
    B1=np.reshape(Bias[1],(nernum,1))
    IW=np.transpose(netW[0])
    LW=np.transpose(netW[1])
    bias1l.append(np.transpose(np.min(np.transpose(B0))))
    bias1u.append(np.transpose(np.max(np.transpose(B0))))
    bias2l.append(np.transpose(np.min(np.transpose(B1))))
    bias2u.append(np.transpose(np.max(np.transpose(B1))))
    IWtotal1l=np.append(IWtotal1l,np.min(IW,0),axis= 0)
    IWtotal1u=np.append(IWtotal1u,np.max(IW,0),axis= 0) 
    LWtotal2l=np.append(LWtotal2l,np.min(LW,0),axis= 0) 
    LWtotal2u=np.append(LWtotal2u,np.max(LW,0),axis= 0)
   
 IWtotal1l=np.reshape(IWtotal1l,(netnum,Ky*optlag))
 IWtotal1u=np.reshape(IWtotal1u,(netnum,Ky*optlag))
 LWtotal2l=np.reshape(LWtotal2l,(netnum,nernum))
 LWtotal2u=np.reshape(LWtotal2u,(netnum,nernum))


# Lower and Upper bounds of neural networks weights based on formulae in
# page 7 of Mohammadi(2019)
 gamal1=np.vstack([(np.min(bias1l)).reshape(-1,1),(np.min(IWtotal1l,0)).reshape(-1,1)])
 gamau1=np.vstack([(np.max(bias1u)).reshape(-1,1),(np.max(IWtotal1u,0)).reshape(-1,1)])
 gamal2=np.vstack([(np.min(bias2l)).reshape(-1,1),(np.min(LWtotal2l,0)).reshape(-1,1)])
 gamau2=np.vstack([(np.max(bias2u)).reshape(-1,1),(np.max(LWtotal2u,0)).reshape(-1,1)])

 GAMAL1=np.kron(np.ones((1,nernum)),gamal1);
 GAMAU1=np.kron(np.ones((1,nernum)),gamau1);
 GAMAL2=np.kron(np.ones((1,nernum)),gamal2);
 GAMAU2=np.kron(np.ones((1,nernum)),gamau2); 

 GAMAL1[GAMAL1<0]=2*GAMAL1[GAMAL1<0];
 GAMAL1[GAMAL1>0]=0.0*GAMAL1[GAMAL1>0];
 GAMAU1[GAMAU1<0]=0.0*GAMAU1[GAMAU1<0];
 GAMAU1[GAMAU1>0]=2*GAMAU1[GAMAU1>0];

 GAMAL2[GAMAL2<0]=2*GAMAL2[GAMAL2<0];
 GAMAL2[GAMAL2>0]=0.0*GAMAL2[GAMAL2>0];
 GAMAU2[GAMAU2<0]=0.0*GAMAU2[GAMAU2<0];
 GAMAU2[GAMAU2>0]=2*GAMAU2[GAMAU2>0];
   
# Obtain phantom functions for nonlinearity test.
 rrmax=1000;
 PvalChipsi0=np.zeros((rrmax,1))
 for rr in range(1,rrmax+1):
    gama=GAMAL1+(GAMAU1-GAMAL1)*np.random.rand(K+1,nernum)
    ner=np.column_stack((np.ones((T,1)),X))@gama
    psigamma=2/(1+np.exp(-2*ner))-1
    
    gama2=GAMAL2+(GAMAU2-GAMAL2)*np.random.rand(nernum+1,nernum)
    ner2=np.column_stack((np.ones((T,1)), psigamma))@gama2
    psigamma2=2/(1+np.exp(-2*ner2))-1


    pca = decomposition.PCA(n_components=0.99-2*math.ceil(np.log(Ky))/100)
    score=pca.fit_transform(psigamma2)
    ndim=len(pca.explained_variance_ratio_)
    scoreX=np.column_stack((X,score[:,0:ndim]))
    (residLWG,yfitTras,betaTras)=nlnTstmols(resid,scoreX)
    sigmrAnn=np.cov(np.transpose(resid))
    sigmauAnn=np.cov(np.transpose(residLWG))
    sAnn=ndim
    tauAnn=K+(T+sAnn+3)/2
    if cresid>1:
     chisqAnn=(T-tauAnn)*(np.log(np.linalg.det(sigmrAnn))-np.log(np.linalg.det(sigmauAnn)))
    else:
      chisqAnn=(T-tauAnn)*(np.log(sigmrAnn)-np.log(sigmauAnn))
    PvalChipsi0[rr-1,0]=1-chi2.cdf(chisqAnn,sAnn*Ky)

 pvalchipsi01=np.sort(PvalChipsi0,axis=0)
 pvalchipsi02= pvalchipsi01*(len(pvalchipsi01)+1-((np.arange(1,len(pvalchipsi01)+1)).reshape(-1,1)))
 pval=np.min(pvalchipsi02)

 # Printing Results
 print()
 print('---- Linearity Test of Univariate and Multivariate Time Series Based on ANN ----' )
 print()
 print('H0: Model is linear,','','PValue of LWG Test','           ',pval) 
 print('---------------------------------------------------------------------------------')
 print('Ref. Mohammadi S.(2019). Neural network for univariate and multivariate'
      ' nonlinearity tests. Stat Anal Data Min: The ASA DataSci Journal.'
      '13:50-70. https://doi.org/10.1002/sam.11441')
 
 return pval




''' Multivariate nonlinearity test

 Discription:
 nonlinearity test for multivariate time series. Output of the code 
 includes following nonliearity tests:

 1- Trasvirta,Lin, and Granger(1993)
 2- Tsay
 3- Keenan

Matlab version of this module has been used in S. Mohammadi(2019).
 
Input: 
 y is a univariate time series in the form column vector.

Output:
 Probability value of the tests. Reject linearity(H0) if pval is
 less than 5%(or any predetermined level of significance).

References:
 1- Mohammadi S. Neural network for univariate and multivariate nonlinearity tests. 
     Stat Anal Data Min: The ASA DataSci Journal. 2019.
     13:50-70.https://doi.org/10.1002/sam.11441
 2- Tsay, R. S. Testing and modeling multivariate threshold models,
     J. Amer. Statist. Assoc. 93 (1998), 1188-1202.
 3- Vavra, M.  Testing for nonlinearity in multivariate stochastic
     processes1, Working paper NBS, 2013, http://www.nbs.sk/en/
     publications-issued-by-the-nbs/working-papers.
   
 Copyright, Shapour Mohammadi 2022.shmohmad@ut.ac.ir'''

#--------------------------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels as stats
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn import decomposition
import math


def nlnTstmols(y,x):
 dimy=np.shape(y)
 if len(dimy)==1:
  cy=1
  y=np.ravel(y)
 else:
  cy=dimy[1]  
 dimx=np.shape(x)
 if len(dimx)==1:
     cx=1
 else:
  cx=dimx[1]
 rx=dimx[0]        
 x=np.column_stack((np.ones((rx,1)), x))
 xT=np.transpose(x)
 betaols=np.zeros((cx+1,cy))
 yfit=np.zeros((rx,cy))
 resmdl=np.zeros((rx,cy))
 for i in range(0,cy):
   betaols[:,i]=np.linalg.inv(xT@x)@(xT@y[:,i])
   yfit[:,i]=x@betaols[:,i]
   resmdl[:,i]=y[:,i]-yfit[:,i]
 return resmdl,yfit,betaols

def nlntstmultv(y):

# Finding optimal lag.
 
 dimy=np.shape(y)
 if len(dimy)==1:
     Ky=1
 else:
     Ky=dimy[1]
 T=dimy[0]    
 aiccriter=np.zeros((5,1))
 for lag in range(1,6):
    xlagmat=lagmat(y,maxlag=lag,trim='None',original='ex')
    xlagmat=xlagmat[lag:T,:]
    yadj=y[lag:T,:]
    (resmdl,yfit,betaols)=nlnTstmols(yadj,xlagmat)
    if Ky>1:
     aiccriter[lag-1,0]=np.log(np.linalg.det((np.transpose(resmdl)@resmdl))/(T-lag))+2*lag/(T-lag)
    else:
      aiccriter[lag-1,0]=np.log((np.transpose(resmdl)@resmdl)/(T-lag))+2*lag/(T-lag)  

 optlag=aiccriter.argmin()+1
 X=lagmat(y,maxlag=optlag,trim='None',original='ex')

 X=X[optlag:T,:]
 yadj=y[optlag:T,:]
 dimX=np.shape(X)
 if len(dimX)==1:
     K=1
 else:
     K=dimX[1]
 T=dimX[0]    
         
 dimy=np.shape(yadj)
 if len(dimy)==1:
     Ky=1
 else:
     Ky=dimy[1]
     
 (resid,fitedKeenan,betaKeenan)=nlnTstmols(yadj,X)

 dimresid=np.shape(resid)
 if len(dimresid)==1:
     cresid=1
 else:
     cresid=dimresid[1]

# Keenan Test(1985)
 CRtermsKeenan=np.zeros((T,1))
 for i in range(0,Ky):
    for j in range(i,Ky):
        CRtermKeenan=fitedKeenan[:,i]*fitedKeenan[:,j]
        CRtermsKeenan=np.column_stack((CRtermsKeenan,CRtermKeenan))

 CRtermsKeenan=np.delete(CRtermsKeenan,0,1)
 (rCRtermsKeenan,colCRtermsKeenan)=np.shape(CRtermsKeenan)
 pca = decomposition.PCA(n_components=0.99-2*math.ceil(np.log(Ky))/100)
 scoreKeenan=pca.fit_transform(CRtermsKeenan)
 ndimKeenan=len(pca.explained_variance_ratio_)
 (residKeenan1,yfitKeenan,betaKeenan)=nlnTstmols(scoreKeenan[:,0:ndimKeenan],X)

 (residKeenan2,yfitKeenan2,betaKeenan2)=nlnTstmols(resid,residKeenan1)
 sigmrKeenan=np.cov(np.transpose(resid))
 sigmauKeenan=np.cov(np.transpose(residKeenan2))
 sKeenan=ndimKeenan
 tauKeenan=K+(cresid+sKeenan+3)/2
 if cresid>1:
  chisqKeenan=(T-tauKeenan)*(np.log(np.linalg.det(sigmrKeenan))-np.log(np.linalg.det(sigmauKeenan)))
 else:
  chisqKeenan=(T-tauKeenan)*(np.log(sigmrKeenan)-np.log(sigmauKeenan))    
 PvalChiKeenan=1-chi2.cdf(chisqKeenan,sKeenan*cresid)


# Multivariate Nonlinearity Tests

# Tsay Test(1986)
 CRterms=np.zeros((T,1))
 for i in range(0,K):
    for j in range(i,K):
        CRterm=X[:,i]*X[:,j]
        CRterms=np.column_stack((CRterms,CRterm))

 CRterms=np.delete(CRterms,0,1)
 (rCRterms,colCRterms)=np.shape(CRterms)
 pca = decomposition.PCA(n_components=0.99-2*math.ceil(np.log(Ky))/100)
 scoreTsay=pca.fit_transform(CRterms)
 ndimTsay=len(pca.explained_variance_ratio_)
 (residTsay1,yfitTsay,betaTsay)=nlnTstmols(scoreTsay[:,0:ndimTsay],X)
 (residTsay2,yfitTsay,betaTsay)=nlnTstmols(resid,residTsay1)
 sigmrTsay=np.cov(np.transpose(resid))
 sigmauTsay=np.cov(np.transpose(residTsay2))
 sTsay=ndimTsay
 tauTsay=K+(cresid+sTsay+3)/2
 if cresid>1:
  chisqTsay=(T-tauTsay)*(np.log(np.linalg.det(sigmrTsay))-np.log(np.linalg.det(sigmauTsay)))
 else:
  chisqTsay=(T-tauTsay)*(np.log(sigmrTsay)-np.log(sigmauTsay))
 PvalChiTsay=1-chi2.cdf(chisqTsay,sTsay*cresid)


# Trasvirta et. al. (1993) test V23(Volterra 2nd and 3rd degree series)
 CRtermsTras2=np.zeros((T,1))
 for i in range(0,K):
    for j in range(i,K):
        CRtermTras2=X[:,i]*X[:,j]
        CRtermsTras2=np.column_stack((CRtermsTras2,CRtermTras2))
        
 CRtermsTras2=np.delete(CRtermsTras2,0,1)

 CRtermsTras3=np.zeros((len(X[:,0]),1))
 for i in range(0,K):
    for j in range(i,K):
        for l in range(j,K):
         CRtermTras3=X[:,i]*X[:,j]*X[:,l]
         CRtermsTras3=np.column_stack((CRtermsTras3,CRtermTras3))

 CRtermsTras3=np.delete(CRtermsTras3,0,1)
 CRtermsTras=np.column_stack((CRtermsTras3,CRtermsTras2))
 (rCRterms,colCRterms)=np.shape(CRtermsTras)
 pca = decomposition.PCA(n_components=0.99-2*math.ceil(np.log(Ky))/100)
 scoreTras=pca.fit_transform(CRtermsTras)
 ndimTras=len(pca.explained_variance_ratio_)
 scoreTrasX=np.column_stack((X,scoreTras[:,0:ndimTras]))
 (residTras,yfitTras,betaTras)=nlnTstmols(resid,scoreTrasX)
 sigmrTras=np.cov(np.transpose(resid))
 sigmauTras=np.cov(np.transpose(residTras))
 sTras=ndimTras
 tauTras=K+(cresid+sTras+3)/2
 if cresid>1:
  chisqTras=(T-tauTras)*(np.log(np.linalg.det(sigmrTras))-np.log(np.linalg.det(sigmauTras)))
 else:
  chisqTras=(T-tauTras)*(np.log(sigmrTras)-np.log(sigmauTras))   
 PvalChiTras=1-chi2.cdf(chisqTras,sTras*cresid)
 
# Printing Results
 print()
 print('---------------- Linearity Test for Multivariate time Series-------------------')
 print()
 print('H0: Model is linear,','','PValue of Keenan Test','           ',PvalChiKeenan) 
 print('H0: Model is linear,','','PValue of Tsay Test','             ',PvalChiTsay) 
 print('H0: Model is linear,','','PValue of Terasvirta et al. Test','',PvalChiTras) 
 print('-------------------------------------------------------------------------------')
 print('Ref. Mohammadi S.(2019). Neural network for univariate and multivariate'
      ' nonlinearity tests. Stat Anal Data Min: The ASA DataSci Journal.'
      '13:50-70. https://doi.org/10.1002/sam.11441')
 
 return PvalChiKeenan,PvalChiTsay,PvalChiTras


''' Univariate nonlinearity test


 Description:
 Nonlinearity test for univariate time series. Output of the code includes
 following nonliearity testS:

 1- Ramsey
 2- Keenan
 3- Trasvirta,Lin, and Granger(1993)
 4- Tsay


Input:
 y is a univariate time series in the column vector form.

Output:
 Probability value of the tests. Reject linearity(H0) if pval is
 less than 5%(or any predetermined level of significance).

References:
 1- Keenan,D. M.(1985). A Tukey nonadditivity-type test for time series
      nonlinearity, Biometrika 72, 39-44.
 2- Mohammadi S.(2019). Neural network for univariate and multivariate
      nonlinearity tests. Stat Anal Data Min: The ASA DataSci Journal.
      13:50-70. https://doi.org/10.1002/sam.11441
 2- Ramsey,J. B.(1969). Tests for specification errors in classical linear
      least squares regression analysis, J. R. Stat. Soc B 31, 350-371.
 3- Terasvirta,T., C. Lin, and C. W. J. Granger(1993), Power of the neural
      network linearity test, J. Time Ser. Anal. 14, 209-220.
 4- Tsay, R. S.(1986). Nonlinearity tests for times series, Biometrika 73,
       4, 61-466.


 Copyright, Shapour Mohammadi 2022.shmohmad@ut.ac.ir'''

#--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels as stats
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from scipy.stats import f

def nlnTstols(y,x):

 x=np.column_stack((np.ones((len(y),1)), x));
 xT=np.transpose(x)
 betaols=np.linalg.inv(xT@x)@(xT@y);
 yfit=x@betaols;
 resmdl=y-yfit;
    
 return resmdl,yfit,betaols
def nlntstuniv(y):
 
 # Finding optimal lag.

 dimy=np.shape(y)
 if len(dimy)==1:
  cy=1
  y=y.reshape(-1,1)
 else:
  cy=dimy[1]
 T=dimy[0] 
 AICCrit=np.zeros((5,1))
 for lag in range(1,6):

  xlagmat=lagmat(y,maxlag=lag,trim='None',original='ex')
  xlagmat=xlagmat[lag:T,:]
  if cy==1:
   yadj=y[lag:T]
  else:
   yadj=y[lag:T,:]
  

  (resmdl,yfit,betaols)=nlnTstols(yadj,xlagmat)

  AICCrit[lag-1,0]=np.log((np.transpose(resmdl)@resmdl)/(T-lag))+2*lag/(T-lag);

 optlag=AICCrit.argmin()+1

 X=lagmat(y,maxlag=optlag,trim='None',original='ex')
 X=X[optlag:T,:]
 if cy==1:
  yadj=y[optlag:T]
 else:
  yadj=y[optlag:T,:]

 (T, K)=np.shape(X)
 (resmdl,yfit,betaols)=nlnTstols(yadj,X)


# Nonlinearity Tests:

# Ramsey(1969) Test
 SSR0=sum(resmdl**2)
 XX=np.column_stack((X,yfit**2))
 XXX=np.column_stack((XX,yfit**3))
 (residRamsey,yfitRamsy,betaols)=nlnTstols(resmdl,XXX)
 SSR1=sum(residRamsey**2)
 FRamsey=((SSR0-SSR1)/(2+K+1))/(SSR1/(T-2*K-2-1))
 PvalRamsey=1-f.cdf(FRamsey,2+K+1,T-2*K-2-1)


#Keenan Test
 (residKeenan,yfit,betaols)=nlnTstols(yfit**2,X)
 (residKeenan2,yfit,betasKeenan)=nlnTstols(resmdl,residKeenan)
 etahat0=betasKeenan[1]
 etahat=etahat0*(sum(residKeenan**2))**0.5
 FKeenan=(etahat**2)*(T-K-2)/(sum(resmdl**2)-etahat**2)
 PvalKeenan=1-f.cdf(FKeenan,1,T-K-2)


# Tsay Test(1986)
 CRterms=np.zeros((T,1))
 for i in range(0,K):
        for j in range(i,K):
            CRterm=(X[:,i]*X[:,j]).reshape((len(X[:,0]),1))
            CRterms=np.column_stack((CRterms, CRterm))
           
 CRterms=np.delete(CRterms,0,1)    
 (resTsay1,yfitTsay,betaTsay)=nlnTstols(CRterms,X)
 (TCRTerms,m)=np.shape(CRterms)
 (resTsay2,yfitTsay,betaTsay)=nlnTstols(resmdl,resTsay1)
 resmdlT=np.transpose(resmdl)
 resTsay1T=np.transpose(resTsay1)
 resTsay2T=np.transpose(resTsay2)
 FTsay=((resmdlT@resTsay1)@(np.linalg.inv(resTsay1T@resTsay1))@(np.transpose((resmdlT@resTsay1)))/m)/((resTsay2T@resTsay2)/(T-m-K-1))
 PvalTsay=1-f.cdf(FTsay,m,T-m-K-1)
 PvalTsayPrint=PvalTsay[0]


# Trasvirta et. al. (1993) test V23(Volterra 2nd and 3rd degree series)
 CRtermsTras2=np.zeros((T,1))
 for i in range(0,K):
    for j in range(i,K):
        CRtermTras2=(X[:,i]*X[:,j]).reshape((len(X[:,0]),1))
        CRtermsTras2=np.column_stack((CRtermsTras2,CRtermTras2))
 CRtermsTras2=np.delete(CRtermsTras2,0,1)    
 CRtermsTras3=np.zeros((len(X[:,0]),1))
 for i in range(0,K):
    for j in range(i,K):
        for l in range(j,K):
            CRtermTras3=(X[:,i]*X[:,j]*X[:,l]).reshape((len(X[:,0]),1))
            CRtermsTras3=np.column_stack((CRtermsTras3,CRtermTras3))         
 CRtermsTras3=np.delete(CRtermsTras3,0,1)
                                                      
 SSR0Tras=sum(resmdl**2)
 TrasCRterms=np.column_stack((CRtermsTras2,CRtermsTras3))
 [TTras,mTras]=np.shape(TrasCRterms);
 TrasCRtermsX=np.column_stack((X,TrasCRterms))                                                      
 (TrasResid,Trasyfit,betaTras)=nlnTstols(resmdl,TrasCRtermsX);
 SSR1Tras=sum(TrasResid**2)
 FTras=((SSR0Tras-SSR1Tras)/(mTras))/(SSR1Tras/(T-mTras-K-1))
 PvalTras=1-f.cdf(FTras,mTras,T-mTras-K-1)

# Printing Results
#  print()
#  print('------------------ Linearity Test of Univariate time Series--------------------')
#  print()
#  print('H0: Model is linear,','','PValue of Ramsey Test','           ',PvalRamsey[0])
#  print('H0: Model is linear,','','PValue of Keenan Test','           ',PvalKeenan[0]) 
#  print('H0: Model is linear,','','PValue of Tsay Test','             ',PvalTsayPrint[0]) 
#  print('H0: Model is linear,','','PValue of Terasvirta et al. Test','',PvalTras[0]) 
#  print('-------------------------------------------------------------------------------')
#  print('Ref. Mohammadi S.(2019). Neural network for univariate and multivariate'
#       ' nonlinearity tests. Stat Anal Data Min: The ASA DataSci Journal.'
#       '13:50-70. https://doi.org/10.1002/sam.11441')

 return PvalRamsey,PvalKeenan,PvalTsayPrint,PvalTras

