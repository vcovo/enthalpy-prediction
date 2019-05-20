import numpy as np
import pandas as pd
import csv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

# Load dataset
df = pd.read_csv('../data/dataset_processed.csv', index_col=0)
target = df['Enthalpy(kcal)']
features = df[df.columns[2:]]

# Define search space
Cs = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
epsilons = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Setup the grid to be searched over
param_grid = dict(C=Cs, epsilon=epsilons)

# Define outer folds
kFolds = KFold(n_splits=10, shuffle=True, random_state=9).split(X=features.values, y=target.values)

# Define inner folds
grid_search = GridSearchCV(SVR(kernel='rbf', gamma='auto'), param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=9),
                           n_jobs=19, verbose=1, scoring='neg_mean_squared_error')

# Open results file and write out headers
out_file = open("../results/grid_search_svr.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ['C', 'epsilon', "r2", "error_ma", "error_ms", "error_rms", "error_mp", "error_max"]
wr.writerow(headers)
out_file.flush()

for index_train, index_test in kFolds:
    # Get train and test splits
    x_train, x_test = features.iloc[index_train].values, features.iloc[index_test].values
    y_train, y_test = target.iloc[index_train].values, target.iloc[index_test].values

    # Apply min max normalization
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Fit
    grid_search.fit(x_train, y_train)

    # Get best params
    best_params = grid_search.best_params_

    # Calculate error metrics
    predictions = grid_search.predict(x_test)
    diff = y_test - predictions
    r2 = r2_score(y_test, predictions)
    error_ma = mean_absolute_error(y_test, predictions)
    error_ms = mean_squared_error(y_test, predictions)
    error_rms = np.sqrt(np.mean(np.square(diff)))
    error_mp = np.mean(abs(np.divide(diff, y_test))) * 100
    error_max = np.amax(np.absolute(diff))

    # Write results
    row = [best_params['C'], best_params['epsilon'], r2,
           error_ma, error_ms, error_rms, error_mp, error_max]
    wr.writerow(row)
    out_file.flush()

out_file.close()
