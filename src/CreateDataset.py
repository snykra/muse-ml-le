import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md
from scipy import stats


class CreateDataset:
    base_dir = ''
    granularity = 0

    def __init__(self, base_dir, granularity):
        self.base_dir = base_dir
        self.granularity = granularity

    def add_listening_effort(self, dataset, snr_values = [20, 15, 10, 5, 0]):
        listening_effort = ['very low', 'low', 'neutral', 'high', 'very high']
        snr = []
        le = []
        label = 0; 
        for i in range(0, len(dataset.index)):
            element = dataset.loc[i, 'Elements']
            if type(element) == str:
                if element.startswith("/Marker/"):
                    label = int(element[-1])
            if label != 0:
                snr.append(snr_values[label - 1])
                le.append(listening_effort[label - 1])
            else:
                snr.append(label)
                le.append(0)
        dataset['label_snr'] = snr
        dataset['label_listening_effort'] = le
        return dataset
                    

    def create_dataset(self, start_time, end_time, cols):
        timestamps = pd.date_range(start_time, end_time, freq=str(self.granularity)+'ms')
        data_table = pd.DataFrame(index=timestamps, columns=cols)
        for col in cols:
            data_table[str(col)] = np.nan #initialize the columns
        return data_table

    def num_sampling(self, dataset, data_table, value_cols, aggregation='avg'):
        for i in range(0, len(data_table.index)):
            relevant_rows = dataset[
                    (dataset['TimeStamp'] >= data_table.index[i]) &
                    (dataset['TimeStamp'] < (data_table.index[i] +
                                            timedelta(milliseconds=self.granularity)))]  
            for col in value_cols:
                # numerical cols which for the EEG data are the brain waves
                # We take the average value
                if len(relevant_rows) > 0:
                    data_table.loc[data_table.index[i], str(col)] = np.average(relevant_rows[col])
                else:
                    data_table.loc[data_table.index[i], str(col)] = np.nan 
        return data_table
    
    def cat_sampling(self, dataset, data_table, label_cols):
        for i in range(0, len(data_table.index)):
            relevant_rows = dataset[
                    (dataset['TimeStamp'] >= data_table.index[i]) &
                    (dataset['TimeStamp'] < (data_table.index[i] +
                                            timedelta(milliseconds=self.granularity)))]  
            for col in label_cols:
                # We put 1 when most value of the labels in relevant rows are 1, else 0
                if len(relevant_rows) > 0:
                    #stats.mode prints out mode as well as counts
                    data_table.loc[data_table.index[i], str(col)] = stats.mode(relevant_rows[col])[0] # so only select the mode, not counts
                else:
                    data_table.loc[data_table.index[i], str(col)] = np.nan 
        return data_table

    # Add numerical data, we assume timestamps in the form of nanoseconds from the epoch
    def add_data(self, file, value_cols, label_cols, aggregation='avg'):
        dataset = pd.read_csv(file, skipinitialspace=True)

        dataset['TimeStamp'] = pd.to_datetime(dataset['TimeStamp'])
        dataset = self.add_listening_effort(dataset) # add features for listening effort
        dataset.dropna(thresh=dataset.shape[1]-10,axis=0, inplace=True) #delete the rows of logs of markers (rows with > col-10 nans)

        # now we initialize the sampled dataset with our granularity
        all_columns = value_cols + label_cols

        data_table = self.create_dataset(min(dataset['TimeStamp']), max(dataset['TimeStamp']), all_columns) #this creates a df named data_table 

        data_table = self.num_sampling(dataset, data_table, value_cols) #add numerical data
        data_table = self.cat_sampling(dataset, data_table, label_cols) # add label data
        return data_table