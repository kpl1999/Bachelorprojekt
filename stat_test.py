import pandas as pd
import scipy.stats as st
import numpy as np



def PowerTest(data1: pd.DataFrame, data2: pd.DataFrame):

    # Power-test - and in extension Cohen's d
    n1, n2 = data1.__len__(), data2.__len__()
    mean1, mean2 = data1.mean(), data2.mean()
    diff_std = data1.var() + data2.var()
    cohens_d = abs((mean1-mean2) / np.sqrt(diff_std /2))
    return  cohens_d

def HypothesisTest(data1, data2):
    # t-test for paired data.
    return st.ttest_rel(data1, data2)[1]



notransbsize10 = pd.read_csv("Results/models_and_accuracy_lstm_notrans_bsize10.csv", sep=",")
baselinebsize10 = pd.read_csv("Results/models_and_accuracy_lstm_baseline_bsize10_nsplit5.csv", sep=",")

# Sort df by split
notransbsize10 = notransbsize10.sort_values(by=["split"])
baselinebsize10 = baselinebsize10.sort_values(by=["split"])

print(notransbsize10["r2_test_score"])
print(baselinebsize10["r2_test_score"])


r2_final = notransbsize10["r2_test_score"]
r2_initial = baselinebsize10["r2_test_score"]

RMSE_final = notransbsize10["rmse_test_score"]
RMSE_initial = baselinebsize10["rmse_test_score"]

cohensd_r2 = PowerTest(r2_final, r2_initial)
print("Cohens d:", cohensd_r2)

pvalue_r2 = HypothesisTest(r2_final, r2_initial)
print("p-value:", pvalue_r2)

cohensd_rmse = PowerTest(RMSE_final, RMSE_initial)
print("Cohens d:", cohensd_rmse)

pvalue_rmse = HypothesisTest(RMSE_final, RMSE_initial)
print("p-value:", pvalue_rmse)



