import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
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

# Define grid search
grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=9),
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
model = model2(input_dim, loss=best_params['loss'], r1=best_params['r1'], l1=best_params['l1'], r2=best_params['r2'],
               l2=best_params['l2'])
model.fit(x_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# save the model
filename = '../models/final_ANN_model.h5'
model.save(filename)
