from ast import Num
import copy
from termios import OCRNL

from CreateDataset import CreateDataset
from OutlierDetection import DistributionBasedOutlierDetection
from FeatureEngineering import PrincipalComponentAnalysis, IndependentComponentAnalysis, NumericalAbstraction, FourierTransformation
from MLModels import MLModels
from util.VisualizeDataset import VisualizeDataset

from pathlib import Path
import os
import pandas as pd
import argparse
import numpy as np
from joblib import dump, load

DataViz = VisualizeDataset()

# granularity = milliseconds per instance 
GRANULARITY = 100

def create_dataset(training=True): 
    
    if training:
        DATA_PATH = Path('./data/')
        RESULTS_PATH = Path('./results/create-dataset/')
        RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    else: 
        DATA_PATH = Path('./testing/data/')
        RESULTS_PATH = Path('./testing/results/create-dataset')
        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    for file in os.scandir(DATA_PATH):
        if (file.name.startswith('.')):
            continue
        
        file_path = file.path
        print(f'Creating dataset for {file_path} with granularity {GRANULARITY}')
        dataset = CreateDataset(file_path, GRANULARITY)

        dataset = dataset.add_data(file_path, ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
        'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
        'Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10',
        'Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10',
        'Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10'], ['label_snr', 'label_listening_effort'], 'avg')

        dataset.to_csv(Path(str(RESULTS_PATH) + '/' + file.name))

def detect_outliers(training=True): 
    if training:
        DATA_PATH = Path('./results/create-dataset/')
        RESULTS_PATH = Path('./results/outlier-free/')
        RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    else: 
        DATA_PATH = Path('./testing/results/create-dataset/')
        RESULTS_PATH = Path('./testing/results/outlier-free/')
        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    OutlierDistr = DistributionBasedOutlierDetection()

    for file in os.scandir(DATA_PATH): # go through all instances of experiments  
            file_path = file.path
            print(f'Going through pipeline for file {file_path}.')
            dataset = pd.read_csv(file_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)

            for col in [c for c in dataset.columns if not 'label' in c]: 
                print(f'Measurement is now: {col}')

                print('Step 1: Outlier detection')

                # we use mixture model as it is used in one paper with n=3. Number of outliers is very low 
                # but measurements are short so this is explainable, also we use brain wave data now
                # in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7728142/pdf/sensors-20-06730.pdf

                dataset = OutlierDistr.mixture_model(dataset, col, 3)
                print('Number of outliers for points with prob < 5e-5 for feature ' + col + ': ' + str(dataset[col+'_mixture'][dataset[col+'_mixture'] < 0.0005].count()))
                
                dataset.loc[dataset[f'{col}_mixture'] < 0.0005, col] = np.nan
                del dataset[col + '_mixture']

                print('Step 2: Imputation')
                #print('Before interpolation, number of nans left should be > 0: ' + str(dataset[col].isna().sum()))
                #print('Also count amount of zeroes:' + str((dataset[col] == 0).sum()))

                dataset[col] = dataset[col].interpolate() #interpolating missing values
                dataset[col] = dataset[col].fillna(method='bfill') # And fill the initial data points if needed

                # check if all nan are filled in
                print('Check, number of nans left should be 0: ' + str(dataset[col].isna().sum()))

            # Step 4: save the file
            #print(dataset.head())
            dataset.to_csv(Path(str(RESULTS_PATH) + '/' + file.name))

def view_pca():
    PCA = PrincipalComponentAnalysis()

    for file in os.scandir('./results/outlier-free/'):
        file_path = file.path
        dataset = pd.read_csv(file_path, index_col=0)
        selected_cols = [c for c in dataset.columns if not 'label' in c]
        pc_values = PCA.determine_pc_explained_variance(dataset, selected_cols)
        
        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_cols) + 1)], y=[pc_values],
                    xlabel='principal component number', ylabel='explained variance',
                    ylim=[0, 1], line_styles=['b-'], algo='PCA')

        break

def get_features(training=True):
    if training:
        DATA_PATH = Path('./results/outlier-free/')
        RESULTS_PATH = Path('./results/features/')
        RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    else:
        DATA_PATH = Path('./testing/results/outlier-free/')
        RESULTS_PATH = Path('./testing/results/features/')
        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    PCA = PrincipalComponentAnalysis()
    ICA = IndependentComponentAnalysis()
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    for file in os.scandir(DATA_PATH):
        file_path = file.path
        print(f'Going through pipeline for file {file_path}.')
        dataset = pd.read_csv(file_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        selected_cols = [c for c in dataset.columns if not 'label' in c]

        n_pcs = 5 # change this based on view_pca result
        dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_cols, n_pcs)
        dataset = ICA.apply_ica(copy.deepcopy(dataset), selected_cols)

        milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

        # Freq and time domain features for ws of 1 sec, 2 sec, and 3 sec
        window_sizes = [int(float(1000)/milliseconds_per_instance), int(float(2000)/milliseconds_per_instance),
        int(float(3000)/milliseconds_per_instance)]
        fs = 100 #sample frequency
    
        for ws in window_sizes:          
            dataset = NumAbs.abstract_numerical(dataset, selected_cols, ws, 
            ['mean', 'std', 'max', 'min', 'median', 'slope'])
        
        # we only do fourier transformation for smallest ws [1 sec]
        dataset = FreqAbs.abstract_frequency(dataset, selected_cols, window_sizes[0], fs)

        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.
        # The percentage of overlap we allow:
        window_overlap = 0.5
        # we do this for the biggest ws
        skip_points = int((1-window_overlap) * window_sizes[-1])
        dataset = dataset.iloc[::skip_points,:]

        # save the file
        dataset.to_csv(Path(str(RESULTS_PATH) + '/' + file.name))
        print(dataset.shape)
    
def ml_all():
    DATA_PATH = Path('./results/features/')

    all_datasets = []
    run = 0
    for file in os.scandir(DATA_PATH):
        file_path = file.path
        dataset = pd.read_csv(file_path, index_col = 0)
        dataset.index = pd.to_datetime(dataset.index)
        dataset = dataset.drop('label_snr', axis=1)
        if (run == 0):
            all_datasets = dataset
        else:
            all_datasets = pd.concat([all_datasets], dataset)
        run = run + 1

    print(all_datasets)
    ML = MLModels(all_datasets, 'label_listening_effort', test_size=0.1)

    gnb, lr, dt, knn, svm = ML.all_models()

    models_path = Path('./models/')
    models_path.mkdir(exist_ok=True, parents=True)

    dump(gnb, './models/gnb.joblib')
    dump(lr, './models/lr.joblib')
    dump(dt, './models/dt.joblib')
    dump(knn, './models/knn.joblib')
    dump(svm, './models/svm.joblib')

    return gnb, lr, dt, knn, svm

def predict(model_type):

    # create_dataset(training=False)
    # detect_outliers(training=False)
    # get_features(training=False)

    DATA_PATH = Path('./testing/results/features/')
    RESULTS_PATH = Path('./testing/results/predictions/')
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    model = load('./models/' + model_type + '.joblib')

    for file in os.scandir(DATA_PATH):
        file_path = file.path
        dataset = pd.read_csv(file_path, index_col = 0)
        dataset.index = pd.to_datetime(dataset.index)


        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.fillna(0, inplace=True)

        dataset = dataset.drop([c for c in dataset.columns if 'label' in c], axis=1)

        y = model.predict(dataset)

        labels = {
            0: '0.0',
            1: 'very low', 
            2: 'low',
            3: 'neutral', 
            4: 'high', 
            5: 'very high'
        }

        y_decoded = []

        for i in range(len(y)):
            y_decoded.append(labels[y[i]])

        dataset['prediction_label'] = y

        dataset['prediction'] = y_decoded

        dataset.to_csv(Path(str(RESULTS_PATH) + '/' + file.name))
        

def main():
    if FLAGS.mode == 'full-run': 
        print("Make sure you've run \'--mode view-pca\' first and determined the number of pca components or this will use a default value of 4")
        create_dataset()
        detect_outliers()
        get_features()
        ml_all()

    elif FLAGS.mode == 'view-pca':
        create_dataset()
        detect_outliers()
        view_pca()

    elif FLAGS.mode == 'get-features':
        get_features()
    
    elif FLAGS.mode == 'ml-all':
        ml_all()
    
    elif FLAGS.mode == 'predict':
        predict(FLAGS.model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full-run', help='Select what function to carry out, or \'full-run\' to do the whole process')
    parser.add_argument('--model', type=str, default='gnb', help='Select which model to use for prediction')

    FLAGS, unparsed = parser.parse_known_args()
    main()