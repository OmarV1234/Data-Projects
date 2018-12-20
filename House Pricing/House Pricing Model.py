# Libraries
import pandas as pd
import numpy as np
import xgboost as xgb

# Data
train_set = pd.read_csv("train_set.csv", index_col="Id")

# Basic info for the data
train_set.info()
train_set.head()

# Create dummies
def StringDummies(train_set):

    # Split into numeric and dummies
    numeric_df = train_set.select_dtypes(exclude="object")

    dummies_df = train_set.select_dtypes(include="object")

    # Create dummy variables
    for x in dummies_df.columns:

        dummy_df = pd.get_dummies(dummies_df[x], prefix=x)

        numeric_df = pd.merge(numeric_df,dummy_df,how='outer',left_index=True, right_index=True)

    # Return the results
    return numeric_df

## Run it
XGB_df = StringDummies(train_set)

## Xgboost initial randomization

def XGBoost_Initialization(XGB_df, iterations = 100):

    # Dmatrix
    Y = XGB_df["SalePrice"]
    X = XGB_df.drop(['SalePrice'], axis=1)

    data_dmatrix = xgb.DMatrix(data=X,label=Y)

    # Set up Hyperparameters and run
    params = {'num_feature': 80, 'eta': 0.3, 'max_depth': 4, 'subsample': 0.5, 'silent':1}

    results = xgb.cv(params = params, metrics = ["rmse", "mae"], num_boost_round = 5, nfold = 5, dtrain = data_dmatrix, verbose_eval = False)

    # Extract the final result
    results = results.iloc[-1,:]
    results = pd.DataFrame(results).transpose()

    # Put the hyperparameters in the dataframe
    results["num_feature"] = 80
    results["eta"] = 0.3
    results["max_depth"] = 4
    results["subsample"] = 0.5
    results["num_boost_round"] = 5
    results["nfold"] = 5

    intial_results_df = results

    # Hyperparameter optimization
    for x in range(iterations):

        # Randomly select the parameters
        np.random.seed()

        # Number of features
        num_feature = int(np.random.normal(loc = 100, scale = 30))

        if num_feature > X.shape[1]:
            num_feature = 100
        elif num_feature < 15:
            num_feature = 100

        # Learning rate
        eta = np.random.normal(loc = 0.5, scale = 0.15)

        if eta <= 0:
            eta = 0.5
        elif eta >= 1:
            eta = 0.5

        # Max depth
        max_depth = int(np.random.normal(loc = 10, scale = 3))

        if max_depth <= 2:
            max_depth = 10

        # Subsample
        subsample = np.random.normal(loc = 0.5, scale = 0.15)

        if subsample <= 0.25:
            subsample = 1
        elif subsample >= 1:
            subsample = 1

        # Number of boosting rounds
        num_boost_round = int(np.random.normal(loc = 10, scale = 3))

        if num_boost_round <= 2:
            num_boost_round = 10

        # nfold
        nfold = int(np.random.normal(loc = 10, scale = 3))

        if nfold <= 2:
            nfold = 10

        params = {'num_feature': num_feature, 'eta': eta, 'max_depth': max_depth, 'subsample': subsample, 'silent':1}

        results = xgb.cv(params = params, metrics = ["rmse", "mae"], num_boost_round = num_boost_round, nfold = nfold, dtrain = data_dmatrix,verbose_eval = False)

        # Extract the final result
        results = results.iloc[-1,:]
        results = pd.DataFrame(results).transpose()

        # Put the hyperparameters in the dataframe
        results["num_feature"] = num_feature
        results["eta"] = eta
        results["max_depth"] = max_depth
        results["subsample"] = subsample
        results["num_boost_round"] = num_boost_round
        results["nfold"] = nfold

        # Join to the previous results
        intial_results_df = intial_results_df.append(results,ignore_index=True)

    # Sort and return
    intial_results_df = intial_results_df.sort_values(by=["test-rmse-mean"])
    intial_results_df = intial_results_df.reset_index(drop=True)

    return intial_results_df

# Run it
initial_results = XGBoost_Initialization(XGB_df, iterations = 500)

# Write it down for later
initial_results.to_csv("Initial_results.csv",index=False)

# Read the data
initial_results = pd.read_csv("Initial_results.csv")

initial_results = initial_results.sort_values(by=["test-rmse-mean"])
initial_results = initial_results.reset_index(drop=True)

# Genetic Algorithm for Hyperparameter Optimization

def GeneticAlgorithm(initial_results, XGB_df ,iterations = 100, top_result_percentage = 0.3):

    # Dmatrix
    Y = XGB_df["SalePrice"]
    X = XGB_df.drop(['SalePrice'], axis=1)

    data_dmatrix = xgb.DMatrix(data=X,label=Y)


    # First, get the number of random elements for mutations in the algorithm
    random_elements = int(initial_results.shape[0]*0.3)

    # Now get those elements for each hyperparameter

    # Number of features
    # First, get the number of random elements, then change to integer, then check that results aren't too low or too big

    num_feature = np.random.normal(loc = 100, scale = 30, size = random_elements)
    num_feature = num_feature.astype(int)
    num_feature = np.where( (num_feature < 20) | (num_feature > X.shape[1]) , 100,num_feature)

    # Learning rate
    eta = np.random.normal(loc = 0.5, scale = 0.15, size = random_elements)

    eta = np.where( (eta <= 0) | (eta >= 1) , 0.5 ,eta)

    # Max depth
    max_depth = np.random.normal(loc = 10, scale = 3, size = random_elements)
    max_depth = max_depth.astype(int)
    max_depth = np.where(max_depth <= 2,10 , max_depth)

    # Subsample
    subsample = np.random.normal(loc = 0.5, scale = 0.15)

    subsample = np.where( (subsample <= 0.25) | (subsample >= 1) , 1 ,subsample)

    # Number of boosting rounds
    num_boost_round = np.random.normal(loc = 10, scale = 3, size = random_elements)
    num_boost_round = num_boost_round.astype(int)
    num_boost_round = np.where( num_boost_round <= 2 , 10 ,num_boost_round)

    # nfold
    nfold = np.random.normal(loc = 10, scale = 3, size = random_elements)
    nfold = nfold.astype(int)
    nfold = np.where( nfold <= 2 , 10 , nfold)

    # Create a mutation dataframe

    mutation_dataframe = pd.DataFrame({'num_feature':num_feature, 'eta':eta, 'max_depth':max_depth, 'subsample':subsample,'num_boost_round':num_boost_round,'nfold':nfold})

    # Merge the dataframes
    hyperparameter_df = initial_results.drop(['test-mae-mean','test-mae-std','test-rmse-std','train-mae-mean','train-mae-std','train-rmse-mean', 'train-rmse-std'], axis=1)

    hyperparameter_df = hyperparameter_df.append(mutation_dataframe, ignore_index=True)

    # Get the hyperparameters

    num_feature = np.random.choice(hyperparameter_df["num_feature"])
    eta = np.random.choice(hyperparameter_df["eta"])
    max_depth = np.random.choice(hyperparameter_df["max_depth"])
    subsample = np.random.choice(hyperparameter_df["subsample"])
    num_boost_round = np.random.choice(hyperparameter_df["num_boost_round"])
    nfold = np.random.choice(hyperparameter_df["nfold"])

    # Run the model
    params = {'num_feature': num_feature, 'eta': eta, 'max_depth': max_depth, 'subsample': subsample, 'silent':1}

    results = xgb.cv(params = params, metrics = ["rmse", "mae"], num_boost_round = num_boost_round, nfold = nfold, dtrain = data_dmatrix,verbose_eval = False)

    # Extract the final result
    results = results.iloc[-1,:]
    results = pd.DataFrame(results).transpose()

    # Put the hyperparameters in the dataframe
    results["num_feature"] = num_feature
    results["eta"] = eta
    results["max_depth"] = max_depth
    results["subsample"] = subsample
    results["num_boost_round"] = num_boost_round
    results["nfold"] = nfold

    intial_results_df = results

    # Hyperparameter optimization
    for x in range(iterations):

        ## Randomly select the parameters
        np.random.seed()

        # Select the hyperparameters
        num_feature = np.random.choice(hyperparameter_df["num_feature"])
        eta = np.random.choice(hyperparameter_df["eta"])
        max_depth = np.random.choice(hyperparameter_df["max_depth"])
        subsample = np.random.choice(hyperparameter_df["subsample"])
        num_boost_round = np.random.choice(hyperparameter_df["num_boost_round"])
        nfold = np.random.choice(hyperparameter_df["nfold"])

        # Run the model
        params = {'num_feature': num_feature, 'eta': eta, 'max_depth': max_depth, 'subsample': subsample, 'silent':1}

        results = xgb.cv(params = params, metrics = ["rmse", "mae"], num_boost_round = num_boost_round, nfold = nfold, dtrain = data_dmatrix,verbose_eval = False)

        # Extract the final result
        results = results.iloc[-1,:]
        results = pd.DataFrame(results).transpose()

        # Put the hyperparameters in the dataframe
        results["num_feature"] = num_feature
        results["eta"] = eta
        results["max_depth"] = max_depth
        results["subsample"] = subsample
        results["num_boost_round"] = num_boost_round
        results["nfold"] = nfold

        # Join to the previous results
        intial_results_df = intial_results_df.append(results,ignore_index=True)

    # Sort and return
    intial_results_df = intial_results_df.sort_values(by=["test-rmse-mean"])
    intial_results_df = intial_results_df.reset_index(drop=True)

    return intial_results_df

# Run it
GeneticAlgorithm(initial_results, XGB_df ,iterations = 500, top_result_percentage = 0.3)
