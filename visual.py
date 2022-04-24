import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Visualize the predictions of initial model and the final model

# LSTM 
notransbsize10 = pd.read_csv("Results/models_and_accuracy_lstm_notrans_bsize10.csv", sep=",")
ytest = notransbsize10["y_test"][0].replace("[", "").replace("]", "").replace("\n", "").strip().split(" ")
ypred = notransbsize10["y_pred"][0].replace("[", "").replace("]", "").replace("\n", "").strip().split(" ")

# Create unique list of ytest
ytest_unique = range(8)

ytest_, ypred_, xs = [], [], []

# Loop over rows in notransbsize10
for i in range(notransbsize10.shape[0]):
    row = notransbsize10.loc[i]
    ytest = row["y_test"].replace("[", "").replace("]", "").replace("\n", "").strip().split(" ")
    ypred = row["y_pred"].replace("[", "").replace("]", "").replace("\n", "").strip()
    ypred = re.sub(r'(\s+)', ' ', ypred).strip().split(" ")

    # Convert ytest and ypred to float
    ytest = [float(i) for i in ytest]
    ypred = [float(i) for i in ypred]

    ytest_ += ytest
    ypred_ += ypred


df = pd.DataFrame(columns=["ytest", "ypred", "xs"])

# add ytest and ypred to dataframe
df["ytest"] = ytest_
df["ypred"] = ypred_
# df["xs"] = xs

preds_, true_, xs_ = [], [], []
for i in ytest_unique:
    
    preds_.append(df[df["ytest"] == i]["ypred"])
    true_.append(df[df["ytest"] == i]["ytest"] + 1)

# list of 7 beautiful colours
palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#ff5da9"]

plt.boxplot(preds_, labels = ytest_unique)
for true, val, c in zip(true_, preds_, palette): 
    # print(x, val, c)
    plt.scatter(true, val, color=c, alpha=0.4)

plt.scatter([1,2, 3, 4, 5, 6, 7, 8],  [0,1, 2, 3, 4, 5, 6, 7], color="black", alpha=0.6)

# plt.title("Confidence Intervals for predictions - Model LSTM no_trans bsize10")
plt.xlabel("True Values")
plt.ylabel("Predictions")
# set y lim
plt.ylim((-0.5,8))
plt.show()



########################################
 # LSTM baseline
baselinebsize10 = pd.read_csv("Results/models_and_accuracy_lstm_baseline_bsize10_nsplit5.csv", sep=",")

# Create unique list of ytest
ytest_unique = range(8)

ytest_, ypred_, xs = [], [], []


# Loop over rows in notransbsize10
for i in range(baselinebsize10.shape[0]):
    row = baselinebsize10.loc[i]
    ytest = row["ytest"].replace("[", "").replace("]", "").replace("\n", "").strip().split(" ")
    ypred = row["y_pred"].replace("[", "").replace("]", "").replace("\n", "").strip()
    ypred = re.sub(r'(\s+)', ' ', ypred).strip().split(" ")

    # Convert ytest and ypred to float
    ytest = [float(i) for i in ytest]
    ypred = [float(i) for i in ypred]

    ytest_ += ytest
    ypred_ += ypred
    # xs += np.random.normal(i + 1, 0.04, size=len(ytest)).tolist()


df = pd.DataFrame(columns=["ytest", "ypred", "xs"])

# add ytest and ypred to dataframe
df["ytest"] = ytest_
df["ypred"] = ypred_
# df["xs"] = xs

preds_, true_, xs_ = [], [], []
for i in ytest_unique:
    
    preds_.append(df[df["ytest"] == i]["ypred"])
    true_.append(df[df["ytest"] == i]["ytest"] + 1)
    # xs_.append(df[df["ytest"] == i]["xs"])

# list of 7 beautiful colours
palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#ff5da9"]

plt.boxplot(preds_, labels = ytest_unique)
for true, val, c in zip(true_, preds_, palette): 
    # print(x, val, c)
    plt.scatter(true, val, color=c, alpha=0.4)

# plot true values
plt.scatter([1,2, 3, 4, 5, 6, 7, 8],  [0,1, 2, 3, 4, 5, 6, 7], color="black", alpha=0.6)


plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.legend(loc='upper left')
plt.ylim((-0.5,8))
plt.show()