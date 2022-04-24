from scipy import special, stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import math


class Transformer():
    '''
    This class can be used for applying transformations and inverse transformations to the target variable.
    Supported transformations:
        -> square root
        -> log
        -> box cox
    '''
    def __init__(self):
        self._lambda = 0.1 # init lambda -> required for box cox inverse transformation
    
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
            # data_transformed, self._lambda = stats.boxcox(data_in)
            data_transformed = special.boxcox(data_in, self._lambda)
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


# Set seed for reproducibility
SEED = 220799
np.random.seed(SEED)

# iterate over different transformations and train model plus get error
transformations_list = [
    "no_transformation",
    "boxcox",
    "log",
    "square_root"
]

# Data load
df_total = pd.read_csv("data_model_ML.csv", sep=";")
df_total = df_total.iloc[: , 1:]
df_total = df_total.drop(columns=["TEMP_max", "TEMP_min", "TEMP_mean", "TEMP_var", 
                                "HR_max", "HR_min", "HR_mean", "HR_var", 
                                "EDA_max", "EDA_min", "EDA_mean", "EDA_var"]) # To create data with only one variable

df_total = df_total.drop(columns=["BVP_min", "BVP_max", "BVP_mean", "BVP_median", "BVP_var"])

# "BVP_min", "BVP_max", "BVP_mean", "BVP_median", "BVP_var"

# Load target data
TargetData = pd.read_csv("TargetData.csv", sep =";")
TargetData = TargetData.sort_values(['particpant_ID','Round','Phase'])
frus = TargetData["frustrated"]


# Intialize
X = df_total.to_numpy()
y = frus.to_numpy()


models_and_accuracy = []

n_splits = 5

# Cross validation
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # use MinMaxScaler to scale training data and test data
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results_dict = {}
    for transformation in transformations_list:
        
        # Set copy of y_train and y_test since we need to manipulate them for log and boxcox
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
       
        y_train_transformed = transformer.apply_transformation(y_train_copy, transformation)
        
        # create linear regression model and train
        reg = LinearRegression().fit(X_train, y_train_transformed)

        # create predictions on test set
        preds = reg.predict(X_test)
        
        # transform back
        preds = np.round(transformer.apply_inverse_transformation(preds, transformation),4)

        # get mse and r2
        r2 = round(r2_score(y_test, preds),3)
        # r2 = round(r2_score(y_test, preds), 5)
        rmse = round(math.sqrt(mean_squared_error(y_test, preds)), 3)

        # add to results dict
        results_dict[transformation] = [r2, rmse]
        
        # store in results dict
        # results_dict[transformation] = [r2, rmse]

        # store in results dict
        models_and_accuracy.append({'split': i, 'r2_test_score': r2, 'rmse_test_score': rmse, 'transformation': transformation, "y_test": y_test, "y_pred": preds})


    # df_results = pd.DataFrame.from_dict(results_dict, orient="index", columns=["R2-Score", "RMSE"])


models_and_accuracy = sorted(models_and_accuracy, key=lambda k: k['r2_test_score'], reverse=True)
# Convert to dataframe
df_results = pd.DataFrame(models_and_accuracy, columns=["split", "r2_test_score", "rmse_test_score", "transformation", "y_test", "y_pred"])

# Save as csv
df_results.to_csv("models_and_accuracy_linreg.csv", sep=",")

print()
print("-"*10)
for trans in transformations_list:
    indecies = df_results[df_results["transformation"] == trans]
    print(f"Average R-squared score for {trans} transformation: ", np.round(np.mean(indecies["r2_test_score"]),3))
    print(f"Average RMSE score for {trans} transformation: ", np.round(np.mean(indecies["rmse_test_score"]),3))
    print("-"*10)



