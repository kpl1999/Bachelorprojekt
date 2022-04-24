import pandas as pd
import os



dir = "Data_nicklas/targets"


target = sorted(os.listdir(dir))

df_full = pd.DataFrame()

for i in target:
    if not i.startswith("."):
        if "delta" not in i:
            df = pd.read_excel(dir + "/" + i)
            df['Round'] = i[6]
            df['Phase'] = i[-6]
            df_full = pd.concat([df_full, df], axis=0)

df_full = df_full.sort_values(['E4_nr','Round','Phase'])
df_full.to_csv("TargetData.csv", sep=";")
