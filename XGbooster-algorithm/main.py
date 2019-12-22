import time
start_time = time.time()
import joblib
import warnings
warnings.filterwarnings("ignore")
#%load process.py
import pandas as pd
import numpy as np
# np.set_printoptions(suppress=True)
import sys
import argparse

expid=2222
n=10
hayrows=1000#100000
# needlerows=150000
hayfill=99
md=20
nr=200


infinityfill=999999
nafill=0

filename='modeltest1'+str(expid)
##############################################################

hay=pd.read_csv('200k.haystack',low_memory=False,nrows=hayrows)

haynames=hay.index

needle=pd.read_csv('203.needle',nrows=2000)# ,low_memory=False,nrows=needlerows) #read in the actual needles from the desc

needlenames=needle.index

needleflag=[1]*len(needlenames)
hayflag=[0]*len(haynames)

hay.columns=needle.columns
# print(hay.head())
train=needle.append([hay])
# print(needle.head())
# print(train.tail(),"tail")
# print(hay.shape)
# print(needle.shape)
# print(train.shape)
# exit(0)
del hay
#del needle
train=train.reset_index(drop=True)

#hayneedlenames=train['Name']
#train=train.drop('Name',axis=1)

######### cleaning
import re
train=train.replace({'Infinity': infinityfill}, regex=True)

train=train.fillna(nafill)
    # train=train.replace({'#NAME?': '9999'}, regex=True)
    # train=train.replace({'9999': 9999}, regex=True)
    #train['row']=train.index.values #adds a new column which is just the order of stuff-- this is the BAD thing that fucks up the data

X=train.copy(deep=True)

# del train

y=pd.DataFrame(needleflag)
y1=pd.DataFrame(hayflag)
y=y.append([y1])
y=y.reset_index(drop=True)
# y=y.rename(columns={0:'assay'}, inplace=True)
y.rename(columns={0:'assay'}, inplace=True)

# height = hayrows
# width = 1
# z = pd.DataFrame(hayfill, index=range(height), columns=range(width))
# z.columns = ['assay']

#################################################### start my code ###########################################################

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.3)
selector.fit(X) ## pls consider this line, this line is extracting feature columns seems to be nessesary.

X=X[X.columns[selector.get_support(indices=True)]] #this is the data, but without the weights-- IT DROPS A BUNCH OF THE COLUMNS


import time
import xgboost as xgb 
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import heapq
import random
from sklearn.metrics import classification_report

 
 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
y=pd.DataFrame(y)
#y.columns=['target']
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score

sc=StandardScaler()
names=list(X.columns)



X=sc.fit_transform(X) ## this line is changing X contents . this line was issue that produce issue.
# X=pd.DataFrame(X)
# X.columns = names

X=pd.DataFrame(train)
drop_list = list(set(train.columns)-set(names)) ## by this line, I can get columns that seems to be neccessary.

for col in drop_list:
    train.drop(col,axis=1,inplace=True)
    pass

train.columns=names

X = train.reset_index(drop=True) ## X contents was already changed. so I changed all value to original using this line. X value was same as train. but 
## X was changed by line 113  this produce issue. 

#################################################### end my code ###########################################################


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)

# ytrainsid=y_train['SID']
# ytrainsid=pd.DataFrame(ytrainsid)

# ytestsid=y_test['SID']
# ytestsid=pd.DataFrame(ytestsid)



# y_train=y_train.drop('SID', axis=1)
# y_test=y_test.drop('SID', axis=1)

#row_number=X_test.index.values.astype(int) #this is the virulent code

########################################################################################################### END of my edits part 1

###this is the part that is in teh basic xgb



dtrain = xgb.DMatrix(data=X_train.values,feature_names=X_train.columns,label=y_train.values ) 
dtrain_test = xgb.DMatrix(data=X_train.values,feature_names=X_train.columns) 
dtest = xgb.DMatrix(data=X_test.values,feature_names=X_test.columns)

# np.set_printoptions(suppress=True)
 
del X_train
del X_test
del X


part1= time.time() - start_time


start_time = time.time()
file=str(md)+'-'+str(nr)+'-'+str(n)+'n'+str(hayrows)+'for-model-gen-for-first-use'


def create_individual():
    """creates random xgboost  models"""
    # import xgboost as xgb
    param ={ 
            'max_depth': int(random.randrange(1,int(md) , step= 1)),#should be 20
            'alpha':10,
            # 'num_round':int(random.randrange(1, int(nr))),#should be 400
            # 'num_parallel_tree':int(random.randrange(1, int(nr))),#should be 400
            # 'child_wght' :round(random.uniform(0.01, 10.0), 2),
            'learning_rate':round(random.uniform(0.01, 1), 1),
            'colsample_bytree':round(random.uniform(0.01, 1.0), 2),
            # 'tree_method':'gpu_hist',
            # 'n_gpus': -1,
            'subsample': 0.8,
            'objective': 'reg:linear',
            'seed':123,
            # 'objective': 'multi:softprob',
            }
    return param

import random
#n=100 #number of models to be created
stone_age=[create_individual() for i in range(n)]

asdf=y_test['assay']

asdf=pd.DataFrame(asdf)
import datetime
import matplotlib.pyplot as plt
def fraud_fitness(model ,dtrain,dtest,y_test):
        params=create_individual()
        # num_round=param['num_round']
        num_round = int(random.randrange(1, int(nr)))
        # print(num_round,"num_round")
        # num_boost_round=random.randrange(1, 10)
        # params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
        bst= xgb.train(dtrain=dtrain, params=params,num_boost_round=num_round)
        # cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3,
        #             num_boost_round=150,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
        # print(cv_results)
        # print((cv_results["test-rmse-mean"]).tail(1),"tail")
        # print(bst)
        # exit(0)
        # return
        # joblib.dump(bst, filename+str(count)+'.mod3') #my addition-- dumps the model for each iteration
        # bst = joblib.load('10.mod3')
        pred=bst.predict(dtest,ntree_limit=bst.best_iteration)
        df = pd.DataFrame()
        df['pred'] = pred
        df['pred'] = [ 0 if item<0.6 else 1 for item in pred]
        df['label'] = y_test['assay'].values
        # evaluate predictions
        # accuracy = accuracy_score(df['pred'], df['label'])
        # print(accuracy,"accuracy for test")
        # pred=bst.predict(dtrain_test,ntree_limit=bst.best_iteration)
        # print(pred)
        # exit(0)
        # pred = [ 0 if item<0.6 else 1 for item in pred]
        # labels = y_train['assay'].values
        # accuracy = accuracy_score(pred, labels)
        # print(accuracy,"accuracy for train")
        # return
        #        print(pred)
        #        print(pred.shape)

        DT = datetime.datetime.now()
        nam=str(DT.microsecond)
        asdf[nam]=pred
       
        
        from sklearn.metrics  import mean_squared_error
        #         print('y_test')
        #         print(y_test)
        #         print('pred')    
        #         print(pred)    
        #        global predtest
        #        predtest=pred

        #fixing b to be like i need it to be
        datain=pd.DataFrame(y_test)
        datain=datain.reset_index(drop=True)
        datain['pred']=pd.DataFrame(pred)
        dataout=datain.copy(deep=True)
        for col in dataout.columns:
            dataout[col].values[:] = 0
        threshold=.9
        index=datain.columns[1]
        dataout[index]=np.where(datain[index]>threshold, 1,0)    
        dataout['assay']=datain['assay']
        dataout[index]=np.where(dataout[index]==1,dataout['assay']+100,0)+np.where(dataout[index]==0,dataout['assay']+200,0)
        tempframe = pd.DataFrame(    [[        0,1,2,3,4    ]])
        tempframe = tempframe.append(        pd.DataFrame(        [[            index,(dataout[index]==101).sum(),(dataout[index]==100).sum(),(dataout[index]==201).sum(),(dataout[index]==200).sum()        ]]        )    )
        tempframe['%ofhits_found']=tempframe[1]/(tempframe[1]+tempframe[3])
        tempframe['poshits_that_are_pos']=tempframe[1]/(tempframe[1]+tempframe[2])
        tempframe['neghits_that_are_neg']=tempframe[4]/(tempframe[3]+tempframe[4])
        tempframe['score']=tempframe['%ofhits_found']+tempframe['poshits_that_are_pos']
        tempframe=tempframe.set_index(0)
        tempframe=tempframe.drop(tempframe.index[0])
        tempframe=tempframe.sort_values(by=['score'])
        #         print(tempframe)
        b=tempframe['score'].item() 
        del tempframe
        del datain 
        del dataout
        return    b

count=1

class generation(object):
    
    def __init__(self,p):
        """here  the  class is initialized with initial(stone age population) . Meaning
        random   neural net arhitectures or  xgb"""
        self.gen=p
        self.perf=[]
    
    def fitness(self):
        global count
        for model in self.gen:
            b=fraud_fitness(model,dtrain,dtest,y_test)
            print('b is'+str(b))
            self.perf.append(b)
            count=count+1
            print(count)
        return self.gen

s=generation(stone_age)

s.fitness()

part2= time.time() - start_time
part2


asdf.to_csv(file+'.csv')

tempperf=pd.DataFrame(s.perf)
tempperf.to_csv(file+'.perf') #this is the mean squared error

f = open( file+'.time', 'w' )
f.write( str(part1)+','+str(part2) )
f.close()