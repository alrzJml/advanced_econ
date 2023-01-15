% % clc
clear
tic
%This code finds paires stocks based on Johanson cointegration test for paires trading
% Written by Shapour Mohammadi
% University of Tehran, shmohmad@ut.ac.ir
testsmpl=126;
tickers2=readcell('symbolsallmarkets.xlsx');
tickers = {'MMM'; 'AES'};
% symbolsnum(1,1)=0;
% for j=1:1
% for i=1:length(tickers)
%     if cell2mat(tickers(i,j))~=0
% symbolsnum(1,j)=symbolsnum(1,j)+1;
% end
% end
% end

cntj=0;
for j=1:1
    cntj=cntj+1;
cnti=0;
for i=1:2
    cnti=cnti+1;
  data = getMarketDataViaYahoo(tickers{i,j},datestr(today-756),datestr(today),'1d');
  if ~isempty(data)
      p0=log(table2array(data(:,6))); 
      Date0=table2array(data(:,1));
      p{i,j}=p0(1:end-testsmpl,1);
      Date{i,j}=Date0(1:end-testsmpl,1);
      ptest{i,j}=p0(end-testsmpl+1:end,1);
      Datetest{i,j}=Date0(end-testsmpl+1:end,1);

  else
      p{i,j}=[]; 
      Date{i,j}=[];
  end
end
end
toc


for j=1:1
for i=1:1
    for k=i+1:2
 
    
    if ~isempty(p{i,j}) && ~isempty(p{k,j})
       pi=p{i,j};
       pk=p{k,j};
       Datei=Date{i,j};
       Datek=Date{k,j};

[comdate,indxi,indxk]=intersect(Datei,Datek);

       ptesti=ptest{i,j};
       ptestk=ptest{k,j};
       Datetesti=Datetest{i,j};
       Datetestk=Datetest{k,j};

[comdatetest,indxtesti,indxtestk]=intersect(Datetesti,Datetestk);


 %Finding optmal lag by AIC criterion
    y=[pi(indxi) pk(indxk)];
    ytest=[ptesti(indxtesti) ptestk(indxtestk)];
    [ry,cy]=size(y);
      maxlag=5;
      if length(y(:,1))>2^2*maxlag+2+10
   for L=1:maxlag
       varmdl=varm(cy,L);
       estmdl=estimate(varmdl,y);
       smdl=summarize(estmdl);
       AICcrit(L,1)=smdl.AIC;
   end

   Lopt=find(AICcrit==min(AICcrit));


% Johansen cointegration test for finding paires
   [H,pvalue,~,~,mles]=jcitest(y,'lags',1,'model','H1');
   if H{1,1}==1
    paires(i,k)=1;
    PairesName(i,1:2)={tickers(i,j) tickers(k,j)};
      BJ2 = mles.r1.paramVals.B;
      c0J2 = mles.r1.paramVals.c0;

      tick1 = tickers(i, j);
      tick2 = tickers(k, j);

      yy = y;

 
% Normalize the cointegrating relation with respect to
% the 1st variable
BJ2n = BJ2(:,1)/BJ2(1,1);
c0J2n = c0J2(1)/BJ2(1,1);
 
    
         cointRinsmpl=y*BJ2n+c0J2n;
         cointRtest=ytest*BJ2n+c0J2n;

         scointR=std(cointRinsmpl);
         mcointR=mean(cointRinsmpl);
         cointR=[cointRinsmpl; cointRtest];
         longs=cointR<=mcointR-2*scointR; 
         shorts=cointR>=mcointR+2*scointR; 
         exitLongs=cointR>=mcointR-1*scointR; 
         exitShorts=cointR<=mcointR+1*scointR; 
         positionsL=zeros(length([comdate;comdatetest]), 2); 
         positionsS=zeros(length([comdate;comdatetest]), 2); 
         positionsS(shorts, :)=repmat([-1 1],[length(find(shorts)) 1]); 
         positionsL(longs,  :)=repmat([1 -1],[length(find(longs)) 1]); 
         positionsL(exitLongs,  :)=zeros(length(find(exitLongs)),2); 
         positionsS(exitShorts,  :)=zeros(length(find(exitShorts)), 2); 
         positions=positionsL+positionsS;
yret=diff(log([y;ytest]));
pnl=positions(1:end-1,1).*yret(:,1)-BJ2n(2)*positions(1:end-1,2).*yret(:,2);
        
        rsuminsmpl=cumsum(pnl(1:end-testsmpl,1));
        rsumtest=cumsum(pnl(end-testsmpl+1:end,1));
        ShrpRatinsmpl=sqrt(252)*mean(pnl(1:end-testsmpl,1))/std(pnl(1:end-testsmpl,1));
        ShrpRatiTest=sqrt(252)*mean(pnl(end-testsmpl+1:end,1))/std(pnl(end-testsmpl+1:end,1));

        figure
        subplot(2,1,1)
        plot(comdatetest,rsumtest)
        title(['Out of Sample Cumulative Return for Pair' ' ' cell2mat(tickers(i,j)) ' ' 'and' ' ' cell2mat(tickers(k,j)) ' ' ...
            'With Sharp Ratio:' ' ' num2str(ShrpRatiTest)])
        subplot(2,1,2)
         plot(comdate(2:end),rsuminsmpl)
        title(['In Sample Cumulative Return for Pair' ' ' cell2mat(tickers(i,j)) ' ' 'and' ' ' cell2mat(tickers(k,j)) ' ' ...
            'With Sharp Ratio:' ' '   num2str(ShrpRatinsmpl)])


 % Plots for signals and prices       
         figure
       subplot(2,1,1)
         plot(comdate,y(:,1),'-b')
       hold on
         plot(comdatetest,ytest(:,1),'-r') 
         plot(comdate,y(:,2),'-k') 
         plot(comdatetest,ytest(:,2),'-r')
         title(['Pair Prices for' ' ' cell2mat(tickers(i,j)) ' '  'and' ' '  cell2mat(tickers(k,j))])

       subplot(2,1,2)
         plot(comdate,cointR(1:length(comdate)),'-b')
         hold on
         plot(comdatetest,cointR(length(comdate)+1:length(comdatetest)+length(comdate)),'-m');

         title(['Cointegrating Relations for' ' '  cell2mat(tickers(i,j)) ' '  'and' ' '  cell2mat(tickers(k,j))])
         hold on
         plot([comdate;comdatetest],(mcointR-2*scointR)*ones(length(cointR),1),'-b')
         plot([comdate;comdatetest],(mcointR+2*scointR)*ones(length(cointR),1),'b')
         plot([comdate;comdatetest],(mcointR-1*scointR)*ones(length(cointR),1),'--g')
         plot([comdate;comdatetest],(mcointR+1*scointR)*ones(length(cointR),1),'--g')
         plot([comdate;comdatetest],(mcointR)*ones(length(cointR),1),'-k')
             
         
   else
   paires(i,k)=0;
   end
      
    end
end
end
end
end
