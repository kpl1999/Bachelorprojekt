from pprint import pprint
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from pprint import pprint

from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold, validation_curve
from scipy import stats, special

#### CONSTANTS ####
SEED = 220799

# Set seed
np.random.seed(SEED)

b_size = 10 # 10, 12

# Define transformations
transformations_list = [
    "no_transformation",
    # "boxcox",
    # "log",
    # "square_root"
]

epochs = 50
n_splits = 5

timesteps = 293 #how many samples/rows/timesteps to look at - X.shape[1]
n_features = 7 # how many predictors/Xs/features we have to predict y - X.shape[2]

LSTM_units = 500
activation_function = "tanh"
optimiser = "adam"
loss = "mse"

###################

# Define model - Regressive LSTM
def model():
    model = Sequential()
    model.add(LSTM(LSTM_units, input_shape=(timesteps, n_features), activation=activation_function, return_sequences=True))
    model.add(LSTM(LSTM_units, activation=activation_function))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimiser)
    model.summary()
    return model

# Function to reshape to 2D array and scale
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

# Transformer Class
class Transformer():


    '''
    This class can be used for applying transformations and inverse transformations to the target variable.
    Supported transformations:
        -> square root
        -> log
        -> box cox
    '''
    def __init__(self):
        self._lambda = 0.5 # init lambda -> required for box cox inverse transformation
    
    def apply_transformation(self, data_in, transform_key):
        '''
        This function applies the transformation according to transformer_key to the provided input.
        
        Args:
            data_in (np.array): Input data to transform
            transform_key (string): Key which transformation to apply (can be: square_root, log, boxcox, no_transformation)
        
        Returns: 
            data_transformed (np.array): The transformed data
        '''
        if transform_key == "no_transformation":
            data_transformed = data_in
        elif transform_key == "square_root":
            data_transformed = np.sqrt(data_in)
        elif transform_key == "log":
            data_transformed = np.log(data_in)
        elif transform_key == "boxcox":
            data_transformed, self._lambda = stats.boxcox(data_in)
        else:
            raise ValueError(f"{transform_key} is an invalid option!")
        
        return data_transformed
    
    def apply_inverse_transformation(self, data_in, transform_key):
        '''
        This function applies the inverse transformation according to transformer_key to the provided input.
        
        Args:
            data_in (np.array): Input data to transform
            transform_key (string): Key which transformation to apply (can be: square_root, log, boxcox, no_transformation)
        
        Returns: 
            data_transformed (np.array): The transformed data
        '''
        if transform_key == "no_transformation":
            data_transformed = data_in
        elif transform_key == "square_root":
            data_transformed = data_in**2
        elif transform_key == "log":
            data_transformed = np.exp(data_in)
        elif transform_key == "boxcox":
            data_transformed = special.inv_boxcox(data_in, self._lambda)
        else:
            raise ValueError(f"{transform_key} is an invalid option!")
        
        return data_transformed


# Load Data
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

# Implement cross validation and use reshape_to_2D_and_back function and save the best model
models_and_accuracy = []


cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
for i, (train, test) in tqdm(enumerate(cv.split(X, y))):
    # Train test split
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    # Print split index
    print()
    print("*"*50)
    print(f"Split {i+1} out of {n_splits}")
    print(f"Total splits: {(n_splits)*len(transformations_list)}")
    print("*"*50)

    Xtrain = reshape_to_2D_and_back(X_train)
    Xtest = reshape_to_2D_and_back(X_test)
    
    # Create transformation loop
    for transformation in transformations_list:

        # Print info
        print("="*50)
        print(f"Transformation: {transformation}")
        print("="*50)

        y_train_copy = y_train.copy()

        transformer = Transformer()

        # +1 for log transformation (cant take 0)
        if transformation == "log":
            y_train_copy = y_train + 1
        # Source: https://discuss.analyticsvidhya.com/t/methods-to-deal-with-zero-values-while-performing-log-transformation-of-variable/2431 

        # +1e-8 for boxcox transformation (cant take 0)
        if transformation == "boxcox":
            y_train_copy = y_train + 1e-8
        # Source: https://community.jmp.com/t5/Discussions/Box-Cox-transformation-with-0-values/td-p/237514

        # Transform y_train
        y_train_transformed = transformer.apply_transformation(y_train_copy, transformation)
        

        # Define Keras Callbacks for training and validation loss. 
        keras_callback = keras.callbacks.CSVLogger(f"split{i}.csv", separator=',', append=True)

        # Source: https://stackoverflow.com/questions/39063676/how-to-boost-a-keras-based-neural-network-using-adaboost 
        # Create a model and apply Adaboost
        ann_estimator = KerasRegressor(build_fn=model, epochs=epochs,
                                        batch_size=b_size, verbose=2,
                                        random_state=SEED, callbacks=[keras_callback],
                                        validation_split=0.2, optimizer=optimiser)
        boosted_ann = AdaBoostRegressor(base_estimator=ann_estimator, random_state=SEED)
        
        # Fit model
        boosted_ann_model = boosted_ann.fit(X_train, y_train_transformed)
        boosted_ann_predict = boosted_ann_model.predict(Xtest)
        
        # transform predicitions back to original scale
        preds = transformer.apply_inverse_transformation(boosted_ann_predict, transformation)

        # get mse and r2
        r2 = round(r2_score(y_test, preds), 5)
        rmse = round(math.sqrt(mean_squared_error(y_test, preds)), 5)

        models_and_accuracy.append({"split": i,
                                    'model': boosted_ann_model,
                                    'r2_test_score': r2,
                                    'rmse_test_score': rmse,
                                    "y_test": y_test,
                                    "y_pred": preds,
                                    "transformation": transformation})
    

# Sort models_and_accuracy by r2_test_score
models_and_accuracy = sorted(models_and_accuracy, key=lambda k: k['r2_test_score'], reverse=True)

# print the best model's r2_test_score, rmse_test_score, ytest and y_pred
print("Best model's r2_test_score:", round(models_and_accuracy[0]['r2_test_score'],3))
print("Best model's rmse_test_score:", round(models_and_accuracy[0]['rmse_test_score'],3))
print("ytest:", models_and_accuracy[0]['y_test'])
print("y_pred:", models_and_accuracy[0]['y_pred'])
print("Best model's transformation:", models_and_accuracy[0]['transformation'])
print("Batch size:", b_size)
print("Epochs:", epochs)
print("Seed:", SEED)

# Convert models_and_accuracy to dataframe
models_and_accuracy_df = pd.DataFrame(models_and_accuracy)
# print(models_and_accuracy_df)

# Save dataframe to csv
print("Saving dataframe to csv")
models_and_accuracy_df.to_csv("models_and_accuracy_lstm.csv", sep=",")

print()
print("-"*10)
for trans in transformations_list:
    indecies = models_and_accuracy_df[models_and_accuracy_df["transformation"] == trans]
    print(f"Average R-squared score for transformation, {trans}: ", np.round(np.mean(indecies["r2_test_score"]),3))
    print(f"Average RMSE score for transformation, {trans}: ", np.round(np.mean(indecies["rmse_test_score"]),3))
    print("-"*10)


