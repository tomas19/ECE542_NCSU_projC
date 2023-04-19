import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def read_data(pathin):
    ''''''
    # Define the header names for the 'x' CSV files
    x_header = ["accx", "accy", "accz", "gyrox", "gyroy", "gyroz"]
    # Define the header names for the 'x_time' CSV files
    x_time_header = ["seconds"]

    # Define the header names for the 'y' CSV files
    y_header = ["class"]

    # Define the header names for the 'y_time' CSV files
    y_time_header = ["seconds"]

    # Create an empty list to store the dataframes
    dct = {}

    filesAll = [x for x in os.listdir(pathin) if x.endswith('.csv')]

    nSubjects = 8

    for i in range(nSubjects):
        files = sorted([x for x in filesAll if x.startswith(f'subject_{i+1:03d}')])
        nSessions = int(len(files)/4) ## 4 files per session
        
        xdataAll = []
        ydataAll = []

        for j in range(nSessions):
            xdata = pd.read_csv(pathin/f'subject_{i+1:03d}_{j+1:02d}__x.csv', header = None, names = x_header)
            xdata['session'] = [j+1]*len(xdata)
            xtime = pd.read_csv(pathin/f'subject_{i+1:03d}_{j+1:02d}__x_time.csv', header = None, names = x_time_header)
            xdata['time'] = xtime['seconds'].values
            xdataAll.append(xdata)

            ydata = pd.read_csv(pathin/f'subject_{i+1:03d}_{j+1:02d}__y.csv', header = None, names = y_header)
            ydata['session'] = [j+1]*len(ydata)
            ytime = pd.read_csv(pathin/f'subject_{i+1:03d}_{j+1:02d}__y_time.csv', header = None, names = y_time_header)
            ydata['time'] = ytime['seconds'].values
            ydataAll.append(ydata)

        dfx = pd.concat(xdataAll)
        dfx.index = range(len(dfx))
        dfy = pd.concat(ydataAll)
        dfy.index = range(len(dfy))

        dct[f'subject_{i+1}_x'] = dfx
        dct[f'subject_{i+1}_y'] = dfy
    return dct

def upsampleData(dct):
    ## Upsample y values, NN interpolation in time column
    dct_ups = {}

    for k in range(int(len(dct.keys())/2)):
        xkey = f'subject_{k+1}_x'
        ykey = f'subject_{k+1}_y'
        dfx = dct[xkey]
        dfy = dct[ykey]

        dfout = []
        
        dfunb = pd.DataFrame(index = dfx['session'].unique(), columns = dfy['class'].unique(), data = 0)
        dfunb['sum'] = [np.nan]*len(dfunb)

        i = 0
        for s in dfx['session'].unique():
            dfxs = dfx[dfx['session'] == s]
            dfys = dfy[dfy['session'] == s]
            distm = cdist(dfxs['time'].values.reshape(len(dfxs), 1), dfys['time'].values.reshape(len(dfys), 1))
            dfxs['class'] = dfys.iloc[distm.argmin(1), 0].values
            dfxs['class_time'] = dfys.iloc[distm.argmin(1), 2].values
            del distm
            dfxs.index = range(i, len(dfxs)+i)
            dfout.append(dfxs)
            i += len(dfxs)
        dfout = pd.concat(dfout, axis = 0)
        dct_ups[f'subject_{k+1}'] = dfout
    
    return dct_ups

def balanceData(dct, threshold = 2):

    dctBal = {f'subject_{k+1}':pd.DataFrame(index = range(1, 9), columns = range(0, 4), data = 0) for k in range(8)}

    for k in dctBal.keys():

        for i in dct[k].index:
            aux = dct[k].loc[i, :]
            aux2 = aux.sort_values(ascending = False)
            aux2.iloc[0] = aux2.iloc[1]

            if aux2.iloc[1] > threshold * aux2.iloc[2]:
                aux2.iloc[0] = threshold * aux2.iloc[2]
                aux2.iloc[1] = threshold * aux2.iloc[2]

            dctBal[k].loc[i, aux2.index] = aux2.values
    return dctBal