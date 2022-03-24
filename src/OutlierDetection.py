from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

class DistributionBasedOutlierDetection:
    # Fits a mixture model towards the data expressed in col and adds a column with the probability
    # of observing the value given the mixture model.
    def mixture_model(self, data_table, col, n_components):
        print('Applying mixture model')
        # Fit a mixture model to our data.
        data = data_table[data_table[col].notnull()][col]
        g = GaussianMixture(n_components, max_iter=100, n_init=1)
        reshaped_data = np.array(data.values.reshape(-1, 1))
        g.fit(reshaped_data)

        # Predict the probabilities
        probs = g.score_samples(reshaped_data)

        # Create the right data frame and concatenate the two.
        data_probs = pd.DataFrame(
            np.power(10, probs), index=data.index, columns=[col+'_mixture'])

        data_table = pd.concat([data_table, data_probs], axis=1)

        return data_table
