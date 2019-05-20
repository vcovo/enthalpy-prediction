import pandas as pd
import pickle
import argparse
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

# Command line argument to determine what model tu use for inference
parser = argparse.ArgumentParser(description='One of SVR or ANN')
parser.add_argument('model_type', type=str, nargs=1)
model_type = parser.parse_args().model_type[0]

# Load the entire set
df = pd.read_csv('../data/dataset_processed.csv', index_col=0)
target = df['Enthalpy(kcal)']
features = df[df.columns[2:]]
test_data = pd.read_csv('../data/octene_isomers.csv')
#test_data = pd.read_csv('../data/nonane_isomers.csv')

# Reduce to the same columns used in features
df_test = test_data[list(features.columns)]
df_test.dropna(inplace=True)

scaler = MinMaxScaler()
scaler.fit(features.values)
x_test = scaler.transform(df_test.values)

# Load the model from disk
model = None
if model_type == 'SVR':
    filename = '../models/final_SVR_model.sav'
    model = pickle.load(open(filename, 'rb'))
if model_type == 'ANN':
    filename = '../models/final_ANN_model.h5'
    model = load_model(filename)

# Infer
predictions = model.predict(x_test).flatten()
print(model_type)
for p in predictions:
    print(p)
