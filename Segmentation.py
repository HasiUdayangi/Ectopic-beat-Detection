from datetime import datetime
import numpy as np
import pandas as pd
import wfdb
from wfdb import rdrecord

startTime = datetime.now()

#MITBIH-
data_all = []

for j in range(len(DS)):
    ecg = DS[j]
    ann = wfdb.rdann(ecg, extension="atr",pn_dir = 'svdb')
    signal, fields = wfdb.rdsamp(ecg, pn_dir = 'svdb')
    signal = signal[:,0]
    #signal = scale(signal)
    data = []
    
    for i, (label, peak) in enumerate(zip(ann.symbol, ann.sample)):
    
        if isinstance(mode, list):
            if np.all([i > 0, i + 1 < len(ann.sample)]):
                left = ann.sample[i - 1] + mode[0]
                right = ann.sample[i + 1] - mode[1]
            else:
                continue
        elif isinstance(mode, int):
            left, right = peak - mode // 2, peak + mode // 2
        else:
            raise Exception("Wrong mode in script beginning")
        data.append(normalization(signal[left:right]))
        
    indices = []
    df = pd.DataFrame()

    for k in range(len(data)):
        df = pd.concat([df, pd.DataFrame(data[k]).T], ignore_index=True).fillna(0)
        if k == "it":
            indices.extend([f"n={i+1}" for i in range(len(data[k]))])
        else:
            indices.append(k)

    df.index = indices
    df.columns = df.columns + 1
    df['label']=pd.Series(ann.symbol)
    
    data_all.append(df)   
    
print (datetime.now() - startTime)

segmentAlldata_df = pd.DataFrame(np.concatenate(data_all))


#MITBIH-LT
data_all2 = []

for j in range(len(data_list2)):
    ecg = data_list2[j]
    ann = wfdb.rdann(ecg, extension="atr",pn_dir = 'ltdb', sampfrom = 0, sampto = 2000000)
    signal, fields = wfdb.rdsamp(ecg, pn_dir = 'ltdb', sampfrom = 0, sampto = 2000000)
    signal = signal[:,0]
    #signal = scale(signal)
    data = []
    
    for i, (label, peak) in enumerate(zip(ann.symbol, ann.sample)):
    
        if isinstance(mode, list):
            if np.all([i > 0, i + 1 < len(ann.sample)]):
                left = ann.sample[i - 1] + mode[0]
                right = ann.sample[i + 1] - mode[1]
            else:
                continue
        elif isinstance(mode, int):
            left, right = peak - mode // 2, peak + mode // 2
        else:
            raise Exception("Wrong mode in script beginning")
        data.append(normalization(signal[left:right]))
        
    indices = []
    df = pd.DataFrame()

    for k in range(len(data)):
        df = pd.concat([df, pd.DataFrame(data[k]).T], ignore_index=True).fillna(0)
        if k == "it":
            indices.extend([f"n={i+1}" for i in range(len(data[k]))])
        else:
            indices.append(k)

    df.index = indices
    df.columns = df.columns + 1
    df['label']=pd.Series(ann.symbol)
    
    data_all2.append(df)   

    
    
