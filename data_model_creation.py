import os
import os.path
import pandas as pd

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
                            df = pd.read_csv(os.path.join(dirpath, filename))
                            df = df.iloc[:,1:]
                            df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d %H:%M:%S')
                            df = df.set_index('time') 
                            df = df.resample('1S').mean() # Down-sampling
                            df = df.dropna()
                            df = df.head(293)
                            df = df.reset_index()
                            df.columns = [f"time_{df.columns[1]}",df.columns[1]]

                            df_full = pd.concat([df_full,df],axis=1)
                           
                df_full['ID'] = str(i[1])
                df_full["Round"] = str(j[-1])
                df_full["Phase"] = str(k)
                df_total = pd.concat([df_total,df_full],axis=0)
                df_full = pd.DataFrame()

df_total = df_total.sort_values(['ID','Round','Phase'])

df_total = df_total.drop(columns=["time_TEMP", "time_EDA", "time_HR", "time_BVP"], axis=1)


# Remove BVP outside accepted range (does automaticcaly when resampling with mean)
# df_total = df_total[df_total['BVP'] < 500]
# df_total = df_total[df_total['BVP'] > -500]

df_total = df_total.reset_index()

df_total.to_csv("data_model.csv", sep=";")

df_total.info()