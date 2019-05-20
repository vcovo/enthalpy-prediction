import numpy as np
import pandas as pd
import csv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
pd.options.mode.chained_assignment = None


def model2(input_dim, loss, r1, l1, r2, l2):
    """
    Returns a two layer artificial neural network model with the number of units
    defined by l1 and l2, and regularization coefficients by r1 and r2.
    """
    model = Sequential()
    model.add(Dense(l1, input_dim=input_dim, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(r1))
    model.add(Dense(l2, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(r2))
    model.add(Dense(1, kernel_initializer='he_uniform'))
    model.compile(loss=loss, optimizer='adam')
    return model


# Load dataset
df = pd.read_csv('../data/dataset_processed.csv', index_col=0)
target = df['Enthalpy(kcal)']
features = df[df.columns[2:]]
input_dim = features.shape[1]

# Define search space
epochs = [1000, 2000, 3000, 4000, 5000]
batch_size = [4, 8, 16, 32, 64, 128, 256, 512]
loss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error']
l1 = [20, 40, 80, 160]
l2 = [20, 40, 80, 160]
r1 = [0.1, 0.2, 0.3, 0.4]
r2 = [0.1, 0.2, 0.3, 0.4]

# Setup the grid to be searched over
param_grid = dict(batch_size=batch_size, epochs=epochs, loss=loss, l1=l1, l2=l2,
                  r1=r1, r2=r2, input_dim=[input_dim])

# Make scikit-learn accepted Keras model
model = KerasRegressor(build_fn=model2, verbose=0)

# Define outer folds
kFolds = KFold(n_splits=10, shuffle=True, random_state=9).split(X=features.values, y=target.values)

# Define inner folds
grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=9),
                           n_jobs=39, verbose=1, scoring='neg_mean_squared_error')

# Open results file and write out headers
out_file = open("../results/grid_search_ann.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ['epochs', 'batch_size', 'loss', 'l1', 'l2', 'd1', 'd2', 'r2'
           'error_ma', 'error_ms', 'error_rms', 'error_mp', 'error_max']
wr.writerow(headers)
out_file.flush()

# Run grid search and write results
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
    row = [best_params['epochs'], best_params['batch_size'], best_params['loss'],
           best_params['l1'], best_params['l2'], best_params['r1'], best_params['r2'],
           r2, error_ma, error_ms, error_rms, error_mp, error_max]
    wr.writerow(row)
    out_file.flush()

out_file.close()
