from sklearn.decomposition import PCA, FastICA
import util.util as util
import numpy as np
import pandas as pd
import scipy.stats as stats

class PrincipalComponentAnalysis:
    def __init__(self):
        self.pca = []

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):
        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)
        # perform the PCA.
        self.pca = PCA(n_components = len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):
        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components = number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        #And add the new ones:
        for comp in range(0, number_comp):
            data_table['pca_' + str(comp + 1)] = new_values[:, comp]

        return data_table
    
class IndependentComponentAnalysis:
    def __init__(self):
        self.ica = []
    
    # Apply a FastICA given the number of components we have selected.
    # We add new ica columns.
    def apply_ica(self, data_table, cols):
        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)

        # perform the FastICA for all components.
        self.ica = FastICA(n_components = len(cols), max_iter = 1000)
        self.ica.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.ica.transform(dt_norm[cols])

        #And add the new ones:
        for comp in range(0, len(cols)):
            data_table['FastICA_' +str(comp + 1)] = new_values[:, comp]

        return data_table

class NumericalAbstraction:
    def get_slope(self, data):      
        times = np.array(range(0, len(data.index)))
        data = data.astype(np.float32)

        # Check for NaN's
        mask = ~np.isnan(data)

        # If we have no data but NaN we return NaN.
        if (len(data[mask]) == 0):
            return np.nan
        # Otherwise we return the slope.
        else:
            slope, _, _, _, _ = stats.linregress(times[mask], data[mask])
            return slope

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
    def aggregate_value(self,data, window_size, aggregation_function):
        window = str(window_size) + 's'
        # Compute the values and return the result.
        if aggregation_function == 'mean':
            return data.rolling(window, min_periods=window_size).mean()
        elif aggregation_function == 'max':
            return data.rolling(window, min_periods=window_size).max()
        elif aggregation_function == 'min':
            return data.rolling(window, min_periods=window_size).min()
        elif aggregation_function == 'median':
            return data.rolling(window, min_periods=window_size).median()
        elif aggregation_function == 'std':
            return data.rolling(window, min_periods=window_size).std()
        elif aggregation_function == 'slope':
            return data.rolling(window, min_periods=window_size).apply(self.get_slope)
        else:
            return np.nan

    def abstract_numerical(self, data_table, cols, window_size, aggregation_function_name):   
        for agg in aggregation_function_name:
            for col in cols:                       
                data_table[col + '_temp_' + agg + '_ws_' + str(window_size)] = self.aggregate_value(data_table[col], window_size, agg)
        return data_table

class FourierTransformation:

    def __init__(self):
        self.temp_list = []
        self.freqs = None

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses
    # the number of samples per second (i.e. Frequency is Hertz of the dataset).
    
    def find_fft_transformation(self, data):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        # print(data.shape)
        transformation = np.fft.rfft(data, len(data))
        # real
        real_ampl = transformation.real
        # max
        max_freq = self.freqs[np.argmax(real_ampl[0:len(real_ampl)])]
        # weigthed
        freq_weigthed = float(np.sum(self.freqs * real_ampl)) / np.sum(real_ampl)

        # pse
        PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
        PSD_pdf = np.divide(PSD, np.sum(PSD))

        # Make sure there are no zeros.
        if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
            pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
        else:
            pse = 0

        real_ampl = np.insert(real_ampl, 0, max_freq)
        real_ampl = np.insert(real_ampl, 0, freq_weigthed)
        row = np.insert(real_ampl, 0, pse)
  
        self.temp_list.append(row)

        return 0

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, columns, window_size, sampling_rate):
        self.freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)

        for col in columns:
            collist = []
            # prepare column names
            collist.append(col + '_max_freq_ws_' + str(window_size))
            collist.append(col + '_freq_weighted_ws_' + str(window_size))
            collist.append(col + '_pse_ws_' + str(window_size))
            collist = collist + [col + '_freq_' + str(freq) + '_Hz_ws_' + str(window_size) for freq in self.freqs]
           
            # rolling statistics to calculate frequencies, per window size. 
            # Pandas Rolling method can only return one aggregation value. 
            # Therefore values are not returned but stored in temp class variable 'temp_list'.
            # note to self! Rolling window_size would be nicer and more logical! In older version windowsize is actually 41. (ws + 1)
            data_table[col].rolling(window_size + 1).apply(self.find_fft_transformation)

            # Pad the missing rows with nans
            frequencies = np.pad(np.array(self.temp_list), ((window_size, 0), (0, 0)), 'constant', constant_values=np.nan)

            # add new freq columns to frame
            data_table[collist] = pd.DataFrame(frequencies, index=data_table.index)

            # reset temp-storage array
            del self.temp_list[:]

        return data_table