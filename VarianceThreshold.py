# -------------------------------------------- #
# ------- VARIANCE THRESHOLD DESCRIPTION ----- #
# -------------------------------------------- #
#                                              #
#   Type Problem:    Feature Selection         #
#   Type Model:      Filter                    #
#   Type Evaluation: Unsupervised              #
#                                              #
# -------------------------------------------- #

import numpy as np


class VarianceThreshold:

    def __init__(self, dataset, num_inputs, variance_max, variance_min):

        # Initialize a list of features
        data_features = [[0] * num_inputs] * len(dataset)

        # Extract all row of features from dataset
        i = 0
        for row in dataset:
            data_features[i] = row[:num_inputs]
            i += 1

        # Define variables
        self.data = data_features
        self.var_max = variance_max
        self.var_min = variance_min
        self.attributes_fitted = []

    def execute(self):

        # Transposed dataset
        x_t = map(list, zip(*self.data))

        attributes = []

        # Iterate over all transposed rows
        for row in x_t:

            # Compute the variance of a row
            var = np.var(row, axis=0)

            # If the var is < or > of defined bounds then keep it
            if self.var_min < var < self.var_max:
                attributes.append([True, var])
            else:
                attributes.append([False, var])

        # Set the fitted attributes
        self.attributes_fitted = attributes

    # Get mapped attributes [index_attr, true/false, variance]
    def get_mapped_attributes(self):

        attributes_map = []

        # Iterate over all attributes
        i = 0
        for attr in self.attributes_fitted:

            value = attr[0]
            var = attr[1]

            attributes_map.append([i, value, var])

            i += 1

        return attributes_map
