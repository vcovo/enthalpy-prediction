# enthalpy_prediction_draft
This is the official repository for the paper "Machine Learning to Predict Standard Enthalpy of Formation of Hydrocarbons". Enthalpy is predicted using both Support Vector Regression (SVR) and Artificial Neural Networks (ANNs).

## Setup 
Anaconda was used to create the virtual environment for this project. Feel free to use one of the following commands to set up the required environment:

From `.yml` file (this preserves all package versions and is thus recommended):  
`conda env create -f enthalpy_prediction.yml`

Conda commands:   
`conda create -n enthalpy_prediction python=3.6`   
`conda activate enthalpy_prediction`   
`conda install pandas keras scikit-learn xlrd`  

If a different method for setting up the environment is preferred please refer to the text of the `.yml` file for the versions of each used package. 

## Overview of files
It must be noted that the files for SVR and ANN are very similar and could be combined in to single files. They were kept separate for this repository in order to reduce ambiguity. 

#### `data/`
* `dataset_complete.csv`: Complete dataset used for the paper
* `dataset_processed.csv`: Dataset filtered to the features used for the models
* `octene_isomers.csv`: The Octene isomer dataset - filtering happens within the `inference.py` script.
* `nonyne_isomers.csv`: The Nonyne isomer dataset - filtering happens within the `inference.py` script.

#### `models/` 
* `final_ANN_model.pkl`: Trained ANN model with the best found combination of hyperparameters. 
{'batch_size': 128, 'epochs': 5000, 'l1': 80, 'l2': 80, 'loss': 'mean_absolute_error', 'r1': 0.1, 'r2': 0.2}
* `final_SVR_model.pkl`: Trained SVR model with the best found combination of hyperparameters. 
{'C': 6000, 'epsilon': 0.15}

#### `results/`
* `grid_search_ann.csv`: The complete results from the grid search for the ANN
* `grid_search_svr.csv`: The complete results from the grid search for the SVR

#### `scripts/`  
* `error_estimation_ann.py`: 10 fold grid search over each of 10 folds of the entire dataset in order to estimate the prediction abilities of an ANN. 
* `error_estimation_svr.py`: 10 fold grid search over each of 10 folds of the entire dataset in order to estimate the prediction abilities of a SVR. 
* `final_model_ann.py`: The script used to generate the final ANN model, `final_ANN_model.pkl`, found through a 10 fold grid search over the entire dataset. 
* `final_model_svr.py`: The script used to generate the final SVR model, `final_SVR_model.pkl`, found through a 10 fold grid search over the entire dataset. 
* `inference.py`: The script in order to infer the enthalpy of the Nonane isomers. Requires a command line argument specifying whether to use the ANN or the SVR. 
* `process_dataset.py`: The script used to create `dataset_processed.csv` from `dataset_complete.csv`. 
* `sensitivity_analysis.py`: The script used in order to run a sensitivity analysis over the final SVR model. 

## Authorship
Code was written by Vincent C.O. van Oudenhoven. Kiran Yalamanchi was responsible for the used data. 

## Acknoledgement 

## License

## Referencing
Still need to add the reference

### BibTex
@article{luong17,
  author  = {Kiran K. Yalamanchi, Vincent C.O. van Oudenhoven, Francesco Tutino, M. Monge-Palacios, Abdulelah Alshehri, S. Mani Sarathy, Xin Gao},
  title   = {},
  journal = {},
  year    = {2019},
}
