import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


def get_sensitivity_scores(model, features, top_n):
    """
    Finds the sensitivity of each feature in features for model. Returns the top_n
    feature names, features_top, alongside the sensitivity values, scores_top.
    """
    # Get just the values of features
    x_train = features.values
    # Apply min max normalization
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    # Find mean and standard deviation of each feature
    x_train_avg = np.mean(x_train, axis=0).reshape(1, -1)
    x_train_std = np.std(x_train, axis=0).reshape(1, -1)
    prediction_mean = model.predict(x_train_avg)

    scores_max = []
    # Iterate over each feature
    for i in range(x_train_avg.shape[1]):
        # Copy x_train_avg
        x_train_i = x_train_avg.copy()
        # Add the standard deviation of i to that column
        x_train_i[:, i] = x_train_i[:, i] + x_train_std[:, i]
        result_i = model.predict(x_train_i)
        # Take the difference and divide by standard deviation
        diff = (result_i - prediction_mean) / x_train_std[:, i]
        scores_max.append(diff.flatten()[0])
    scores_max = np.absolute(scores_max)
    indices_top = np.argsort(scores_max)[-top_n:]
    features_top = features.iloc[:, indices_top].columns
    scores_top = scores_max[indices_top]
    return features_top, scores_top


# Initialize result sheet
df_sensitivity = pd.DataFrame()
n_top = 10

# Load dataset
df = pd.read_csv('../data/dataset_processed.csv', index_col=0)
features = df[df.columns[2:]]

# Load SVR model
filename = '../models/final_SVR_model.sav'
model = pickle.load(open(filename, 'rb'))

features_top, scores_top = get_sensitivity_scores(model, features, n_top)
df_sensitivity['SVR features'] = features_top
df_sensitivity['SVR sensitivity values (abs)'] = scores_top

df_sensitivity.to_csv('../results/sensitivity_analysis.csv')
