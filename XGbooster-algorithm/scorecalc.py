import time
start_time = time.time()

import warnings
warnings.filterwarnings("ignore")
#%load process.py
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import sys
import argparse

#inputfile='20-400-50n150000fill-203_categorical_test.csv'
inputfile=sys.argv[1]

datain=pd.read_csv(inputfile)



threshold=.9




dataout=datain.copy(deep=True)
for col in dataout.columns:
    dataout[col].values[:] = 0


for i in range(2,datain.columns.size): 
    index=datain.columns[i]
    #print(index)
    #print(datain[index])
    dataout[index]=np.where(datain[index]>threshold, 1,0)
    

dataout['Unnamed: 0']=datain['Unnamed: 0']
dataout['assay']=datain['assay']

dataout

for i in range(2,datain.columns.size): 
    index=dataout.columns[i]
    dataout[index]=np.where(dataout[index]==1,dataout['assay']+100,0)+np.where(dataout[index]==0,dataout['assay']+200,0)


dataout

#del tempframe

tempframe = pd.DataFrame(
    [[
        0,1,2,3,4
    ]]
)


for i in range(2,datain.columns.size): 
    index=dataout.columns[i]
    
    tempframe = tempframe.append(

        pd.DataFrame(
        [[
            index,(dataout[index]==101).sum(),(dataout[index]==100).sum(),(dataout[index]==201).sum(),(dataout[index]==200).sum()
        ]]
        )
    )

tempframe['%ofhits_found']=tempframe[1]/(tempframe[1]+tempframe[3])
tempframe['poshits_that_are_pos']=tempframe[1]/(tempframe[1]+tempframe[2])
tempframe['neghits_that_are_neg']=tempframe[4]/(tempframe[3]+tempframe[4])
tempframe['score']=tempframe['%ofhits_found']+tempframe['poshits_that_are_pos']




tempframe=tempframe.set_index(0)
tempframe=tempframe.drop(tempframe.index[0])
tempframe=tempframe.sort_values(by=['score'])

tempframe.to_csv(inputfile+'.scores.csv')
dataout.to_csv(inputfile+'.flags.csv')