from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col

split0 = pd.read_csv("split0.csv", sep=",")
split1 = pd.read_csv("split1.csv", sep=",")
split2 = pd.read_csv("split2.csv", sep=",")
split3 = pd.read_csv("split3.csv", sep=",")
split4 = pd.read_csv("split4.csv", sep=",")

# Loss for each split
loss_split0 = list(split0["loss"])
loss_split1 = list(split1["loss"])
loss_split2 = list(split2["loss"])
loss_split3 = list(split3["loss"])
loss_split4 = list(split4["loss"])

# Validation loss for each split
val_loss_split0 = list(split0["val_loss"])
val_loss_split1 = list(split1["val_loss"])
val_loss_split2 = list(split2["val_loss"])
val_loss_split3 = list(split3["val_loss"])
val_loss_split4 = list(split4["val_loss"])



# Epochs for each split
epochs_split0 = split0["epoch"]
epochs_split1 = split1["epoch"]
epochs_split2 = split2["epoch"]
epochs_split3 = split3["epoch"]
epochs_split4 = split4["epoch"]

# Create extra column in the dataframe in range 0-len(loss_split0)
split0["epoch"] = range(0, len(loss_split0))
split1["epoch"] = range(0, len(loss_split1))
split2["epoch"] = range(0, len(loss_split2))
split3["epoch"] = range(0, len(loss_split3))
split4["epoch"] = range(0, len(loss_split4))

# Get longest of all splits
max_epochs = max(len(loss_split0), len(loss_split1), len(loss_split2), len(loss_split3), len(loss_split4))
print(max_epochs)

print(list(loss_split0)[-1])

# Make all splits the same length
while len(loss_split0) < max_epochs:
    loss_split0.append(loss_split0[-1])
    val_loss_split0.append(val_loss_split0[-1])

while len(loss_split1) < max_epochs:
    loss_split1.append(loss_split1[-1])
    val_loss_split1.append(val_loss_split1[-1])

while len(loss_split2) < max_epochs:
    loss_split2.append(loss_split2[-1])
    val_loss_split2.append(val_loss_split2[-1])

while len(loss_split3) < max_epochs:
    loss_split3.append(loss_split3[-1])
    val_loss_split3.append(val_loss_split3[-1])

while len(loss_split4) < max_epochs:
    loss_split4.append(loss_split4[-1])
    val_loss_split4.append(val_loss_split4[-1])

means_train = []
means_val = []
stds_train = []
stds_val = []

for idx in range(0, max_epochs):
    means_train.append(np.mean([loss_split0[idx], loss_split1[idx], loss_split2[idx], loss_split3[idx], loss_split4[idx]]))
    stds_train.append(np.std([loss_split0[idx], loss_split1[idx], loss_split2[idx], loss_split3[idx], loss_split4[idx]]))

    means_val.append(np.mean([val_loss_split0[idx], val_loss_split1[idx], val_loss_split2[idx], val_loss_split3[idx], val_loss_split4[idx]]))
    stds_val.append(np.std([val_loss_split0[idx], val_loss_split1[idx], val_loss_split2[idx], val_loss_split3[idx], val_loss_split4[idx]]))

idx = 0
print([val_loss_split0[idx], val_loss_split1[idx], val_loss_split2[idx], val_loss_split3[idx], val_loss_split4[idx]])

x = range(max_epochs)

plt.plot(x, means_train, label="loss_split0", color="#1f77b4")
# plot confidence interval
plt.plot(x, means_val, label="val_loss_split0", color="#ff7f0e")

# plot confidence interval
plt.fill_between(x, np.array(means_train) - 2*np.array(stds_train), np.array(means_train) + 2*np.array(stds_train)
                , alpha=0.2, color = "#1f77b4")
plt.fill_between(x, np.array(means_val) - 2*np.array(stds_val), np.array(means_val) + 2*np.array(stds_val)
                    , alpha=0.2, color="#ff7f0e")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()





# plt.plot(split0["epoch"], loss_split2, label="loss_split0")
# plt.plot(split0["epoch"], val_loss_split2, label="val_loss_split0")
# plt.show()


