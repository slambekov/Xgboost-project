chunkrows=1000

import time
start_time = time.time()
import xgboost as xgb

import pandas as  pd

import numpy as np
import pandas as pd
import numpy as np
import joblib
np.set_printoptions(suppress=True)
out_path=""

#X=pd.read_csv('X.csv',low_memory=False)
# reader=pd.read_table('203.desc', low_memory=False,chunksize=10000)
chunk=pd.read_csv('200k.haystack',low_memory=False,nrows=3000) 
name_chunk=[0]*len(chunk.index)
hay=pd.read_csv('203.needle',low_memory=False,nrows=3000)
name_hay=[1]*len(hay.index)
# chunk=pd.read_csv('200k.haystack',low_memory=False,nrows=chunkrows) 
# hay=pd.read_csv('203.needle',low_memory=False,nrows=chunkrows)
# chunk.columns = hay.columns
chunk=chunk.append(hay)

chunk.reset_index(drop=True,inplace=True)

import re
infinityfill=999999
nafill=0
chunk=chunk.replace({'Infinity': infinityfill}, regex=True)
chunk=chunk.fillna(nafill)
name_chunk.extend(name_hay)

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import  StandardScaler
def fit_data(chunk=chunk):
    # print(chunk)
    # exit(0)
    
    # selector = VarianceThreshold(threshold=0.3)
    # selector.fit(chunk)
    
    # chunk=chunk[chunk.columns[selector.get_support(indices=True)]]
    # names=list(chunk.columns)
    # sc=StandardScaler()
    # chunk=sc.fit_transform(chunk)
    # chunk=pd.DataFrame(chunk)
    # chunk.columns=names
    # print(chunk)
    # exit(0)
    return chunk
    pass

# name_chunk=name_chunk.reset_index(drop=True)

# print(name_chunk)
# exit(0)
# chunk=chunk.drop('A',axis=1)
model = joblib.load('model4.model')
droplist=list(set(chunk.columns) - set(model.feature_names))
i=0
for column in droplist:
    chunk=chunk.drop(column,axis=1)
# chunk = fit_data(chunk)
# print(chunk.values)
# print(len(chunk.columns))
# exit(0)
chunk = xgb.DMatrix(data=chunk.values,feature_names=chunk.columns)
output=[]
pred=model.predict(chunk,ntree_limit=model.best_iteration)
# print(pred)
pred = [ 1 if item>0.7 else 0 for item  in pred]

output=pd.DataFrame(output)
output['Name']=name_chunk
output['prediction']=pred
# print(output)
filter = output[output['Name'] == output['prediction']]
print(str(len(filter.values)/len(output.values)*100)+"%")
# output=output.drop('Name',axis=1)

output.to_csv('pred_from_model_use.csv')