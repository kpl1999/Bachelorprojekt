import pandas as pd
import numpy as np
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


# Load dataset
df_total = pd.read_csv("data_model.csv", sep=";")
df_total = df_total.iloc[: , 1:]
df_total_nobvp = df_total.drop(columns=["BVP"])
X = df_total_nobvp

# Labels
TargetData = pd.read_csv("TargetData.csv", sep =";")
TargetData = TargetData.sort_values(['particpant_ID','Round','Phase'])
frus = TargetData["frustrated"]

X = X.to_numpy().reshape(96,293,7)
y = np.array(frus.to_list())

SEED = 220799

# Set seed
np.random.seed(SEED)

# Function to reshape to 2D array
def reshape_to_2D_and_back(X):
    # when splitting it is in 3D - to scale it has to be 2D. 
    # Get shapes of training and test data e.g. how many samples in each
    X_shape = np.shape(X)

    # Reshape to 2D for normalization
    X = X.reshape(X_shape[0]*293,7)

    # Normalization
    Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1

    Xscaler.fit(X)
    X_norm = Xscaler.transform(X)


    # Reshape back to 3D array for LSTM 
    X_norm = X_norm.reshape(X_shape[0],293,7)

    return X_norm


# Loss History
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

timesteps = 293 #how many samples/rows/timesteps to look at - X.shape[1]
n_features = 7 # how many predictors/Xs/features we have to predict y - X.shape[2]

b_size = 10 # 8, 10, 12
epochs = 50

n_splits = 5 # 5, 10


# Implement cross validation and use reshape_to_2D_and_back function and save the best model and the best model's score
models_and_accuracy = []
# Create empty df
df_train_loss = pd.DataFrame()
df_test_loss = pd.DataFrame()

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
for i, (train, test) in tqdm(enumerate(cv.split(X, y))):
    print()
    print("="*50)
    print(f"Split {i+1} out of {n_splits}")
    print("="*50)

    # Train test split
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    Xtrain = reshape_to_2D_and_back(X_train)
    Xtest = reshape_to_2D_and_back(X_test)
    
    # Create a model and apply Adaboost
    # ann_estimator = KerasRegressor(build_fn=model, epochs=epochs, batch_size=b_size, verbose=2, random_state=SEED)
    # boosted_ann = AdaBoostRegressor(base_estimator=ann_estimator, random_state=SEED)
    # boosted_ann_model = boosted_ann.fit(Xtrain, y_train)
    # boosted_ann_predict = boosted_ann_model.predict(Xtest)
    model = Sequential()
    model.add(LSTM(500, input_shape=(timesteps, n_features), activation='tanh', return_sequences=True))
    model.add(LSTM(500, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    history = model.fit(Xtrain, y_train, epochs=epochs,
                        batch_size=b_size, verbose=2,
                        callbacks=[LossHistory()],
                        validation_data=(Xtest, y_test))
    preds = model.predict(Xtest)

    # pprint(vars(history))
    df_train_loss["Epoch"] = history.epoch
    df_train_loss["Loss_",i+1] = history.history["loss"]

    df_test_loss["Epoch"] = history.epoch
    df_test_loss["Loss_",i+1] = history.history["val_loss"]

    # R-sqaure score
    r2_test_score = r2_score(y_test, preds)
    # RMSE score
    rmse_test_score = np.sqrt(mean_squared_error(y_test, preds))

    # Make dictionary and append to models_and_accuracy
    models_and_accuracy.append({"split": i, 'model': model, 'r2_test_score': r2_test_score, 'rmse_test_score': rmse_test_score, "ytest": y_test, "y_pred": preds})


##  Learning Curve plots ##
# Create palette with 5 beautiful colours
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
validation_colour = "#7f7f7f"

# Loop from 2nd to last column
for i in range(1, len(df_train_loss.columns)):
    train_loss = df_train_loss["Loss_",i]
    test_loss = df_test_loss["Loss_",i]
    epoch_list = df_train_loss["Epoch"]

    plt.plot(epoch_list, train_loss, label="Training Loss", color=colors[i-1])
    plt.plot(epoch_list, test_loss, label="Validation Loss", color=validation_colour)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
###########################

## Cleaner and better plot in every way ##
# Palette with 2 beautiful colours
colors = ["#1f77b4", "#ff7f0e"]

# Calculate mean and std of each row
df_train_loss_no_epoch = df_train_loss.drop("Epoch", axis=1)
df_train_loss_no_epoch_mean = df_train_loss_no_epoch.mean(axis=1)
df_train_loss_no_epoch_std = df_train_loss_no_epoch.std(axis=1)

df_test_loss_no_epoch = df_test_loss.drop("Epoch", axis=1)
df_test_loss_no_epoch_mean = df_test_loss_no_epoch.mean(axis=1)
df_test_loss_no_epoch_std = df_test_loss_no_epoch.std(axis=1)

# Plot mean and std
plt.plot(df_train_loss["Epoch"], df_train_loss_no_epoch_mean, label="Training Loss", color=colors[0])
plt.fill_between(df_train_loss["Epoch"], df_train_loss_no_epoch_mean - 2*df_train_loss_no_epoch_std,
                    df_train_loss_no_epoch_mean + 2*df_train_loss_no_epoch_std, alpha=0.2, color=colors[0])

plt.plot(df_test_loss["Epoch"], df_test_loss_no_epoch_mean, label="Validation Loss", color=colors[1])
plt.fill_between(df_test_loss["Epoch"], df_test_loss_no_epoch_mean - 2*df_test_loss_no_epoch_std,
                    df_test_loss_no_epoch_mean + 2*df_test_loss_no_epoch_std, alpha=0.2, color=colors[1])

plt.plot(df_train_loss["Epoch"], df_train_loss_no_epoch_mean + 2*df_train_loss_no_epoch_std, color=colors[0], alpha=0.5)
plt.plot(df_train_loss["Epoch"], df_train_loss_no_epoch_mean - 2*df_train_loss_no_epoch_std, color=colors[0], alpha=0.5)

plt.plot(df_test_loss["Epoch"], df_test_loss_no_epoch_mean + 2*df_test_loss_no_epoch_std, color=colors[1], alpha=0.5)
plt.plot(df_test_loss["Epoch"], df_test_loss_no_epoch_mean - 2*df_test_loss_no_epoch_std, color=colors[1], alpha=0.5)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
##########################################


# Sort models_and_accuracy by r2_test_score
models_and_accuracy = sorted(models_and_accuracy, key=lambda k: k['r2_test_score'], reverse=True)

# print the best model's r2_test_score, rmse_test_score, ytest and y_pred
print("Best model's r2_test_score:", round(models_and_accuracy[0]['r2_test_score'],3))
print("Best model's rmse_test_score:", round(models_and_accuracy[0]['rmse_test_score'],3))
print("ytest:", models_and_accuracy[0]['ytest'])
print("y_pred:", models_and_accuracy[0]['y_pred'])
print("Batch size:", b_size)
print("Epochs:", epochs)


# Average the r2_test_score and rmse_test_score for all the splits
r2_test_score_avg = sum([i['r2_test_score'] for i in models_and_accuracy])/len(models_and_accuracy)
rmse_test_score_avg = sum([i['rmse_test_score'] for i in models_and_accuracy])/len(models_and_accuracy)

# print the average r2_test_score and rmse_test_score
print("Average r2_test_score:", round(r2_test_score_avg,3))
print("Average rmse_test_score:", round(rmse_test_score_avg,3))


# Convert models_and_accuracy to dataframe
models_and_accuracy_df = pd.DataFrame(models_and_accuracy)

# Save dataframe to csv
print("Saving dataframe to csv")
models_and_accuracy_df.to_csv("models_and_accuracy_lstm_baseline.csv", sep=",")
