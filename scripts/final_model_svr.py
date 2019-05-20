import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
pd.options.mode.chained_assignment = None

# Load the entire set
df = pd.read_csv('../data/dataset_processed.csv', index_col=0)
target = df['Enthalpy(kcal)']
features = df[df.columns[2:]]

# Define search space
Cs = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
epsilons = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Setup the grid to be searched over
param_grid = dict(C=Cs, epsilon=epsilons)

# Define grid search
grid_search = GridSearchCV(SVR(kernel='rbf', gamma='auto'), param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=9),
                           n_jobs=19, verbose=1, scoring='neg_mean_squared_error')

# Split data in to features and target
x_train = features.values
y_train = target.values

# Apply min max normalization
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)

# Find best parameters
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

# Retrain model with best parameters found from grid search
best_params = grid_search.best_params_
model = SVR(kernel='rbf', gamma='auto', C=best_params['C'], epsilon=best_params['epsilon'])
model.fit(x_train, y_train)

# save the model
filename = '../models/final_SVR_model.sav'
pickle.dump(model, open(filename, 'wb'))
