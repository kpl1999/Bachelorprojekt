import os
import os.path
import pandas as pd
import numpy as np

#### Create dataset
dir = "data_ordered/"
df_full = pd.DataFrame()
df_total = pd.DataFrame()
ID = os.listdir(dir)

for i in [id for id in ID if id.startswith("A")]:
    if not i.startswith(".DS"):
        
        dir1 = dir + i
        rounds = os.listdir(dir1)
    for j in rounds:
        if not j.startswith("."):
            dir2 = dir1 + "/" + j
            
            for k in range(1,4):
                for dirpath, dirnames, filenames in os.walk(dir2):
                        for filename in [f for f in filenames if f.startswith(f"phase{k}")]:
                            #print(os.path.join(dirpath, filename))
                            df = pd.read_csv(os.path.join(dirpath, filename))
                            col = df.columns[1]
                            df = df.iloc[:,1:]
                            df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d %H:%M:%S')
                            df = df.set_index('time') 
                            df[f'{col}_min'] = df[col].min()
                            df[f'{col}_max'] = df[col].max()
                            df[f'{col}_var'] = df[col].var()
                            df[f'{col}_mean'] = df[col].mean()
                            df[f'{col}_median'] = df[col].median()  
                            df = df.head(1)
                            df = df.reset_index()
                            df = df.drop(["time",col],axis=1)

                            df_full = pd.concat([df_full,df],axis=1)
                           
                df_full['ID'] = str(i[1])
                df_full["Round"] = str(j[-1])
                df_full["Phase"] = str(k)
                df_total = pd.concat([df_total,df_full],axis=0)
                df_full = pd.DataFrame()

df_total = df_total.sort_values(['ID','Round','Phase'])
df_total = df_total.reset_index()

df_total.to_csv("data_model_ML.csv", sep=";")



