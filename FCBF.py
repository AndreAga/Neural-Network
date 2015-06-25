# -------------------------------------------- #
#  FAST CORRELATION-BASED FILTER DESCRIPTION   #
# -------------------------------------------- #
#                                              #
#   Type Problem:    Feature Selection         #
#   Type Model:      Filter                    #
#   Type Evaluation: Supervised                #
#                                              #
# -------------------------------------------- #

from operator import itemgetter
import math


class FCBF:

    def __init__(self, dataset, delta):

        # Get all single values of every feature
        attr_set = []
        data_ = map(list, zip(*dataset))

        for row in data_:
            attr_set.append(set(row))

        # Define variables
        self.dataset = dataset
        self.attr_set = attr_set
        self.delta = delta

        self.su_list = []
        self.su_value = []
        self.su_order = []
        self.attributes_map = []

    # Compute Probability
    # P(X) = Count(X) / Length(Attribute)
    def probability(self, index_x, x):

        count = 0
        for i in range(len(self.dataset)):

            if self.dataset[i][index_x] == x:
                count += 1

        if count != 0:
            return float(count) / float(len(self.dataset))
        else:
            return 0.0

    # Compute Conditional Probability
    # P(X|Y) = P(X INTERSECT Y) / P(Y)
    def conditional_probability(self, index_x, x, index_y, y):

        p_xy = 0
        p_y = 0

        for i in range(len(self.dataset)):

            if self.dataset[i][index_y] == y:
                p_y += 1

                if self.dataset[i][index_x] == x:
                    p_xy += 1

        if p_y != 0:
            return float(p_xy) / float(p_y)
        else:
            return 0.0

    # Compute Conditional Entropy
    # H(X|Y) = - SUM[P(Y) * SUM[P(X,Y) - (log(P(X,Y))/log2)]]
    def conditional_entropy(self, x, y):

        values_x = self.attr_set[x]
        values_y = self.attr_set[y]

        summation_1 = 0
        for y in range(len(values_y)):

            p_y = self.probability(y, list(values_y)[y])

            summation_2 = 0
            for x in range(len(values_x)):

                p_xy = self.conditional_probability(x, list(values_x)[x], y, list(values_y)[y])

                if p_xy != 0.0:
                    summation_2 += p_xy * math.log(p_xy) / math.log(2.0)

            summation_1 += p_y * summation_2

        return -summation_1

    # Compute Entropy
    # H(X) = - SUM[P(x) * (log(x)/log2)]
    def entropy(self, index_x):

        x_values = self.attr_set[index_x]

        summation = 0
        for x in range(len(x_values)):

            p_x = self.probability(index_x, list(x_values)[x])

            if p_x != 0.0:
                summation += p_x * (math.log(p_x) / math.log(2.0))

        return -summation

    # Compute Information Gain
    # IG(X,Y) = H(X) - H(X|Y)
    def information_gain(self, x, y):

        h_x = self.entropy(x)
        h_xy = self.conditional_entropy(x, y)

        return h_x - h_xy

    # Compute SU Value
    # SU(X,Y) = 2 * [IG(X|Y) / H(X) + H(Y)]
    def su(self, x, y):

        ig_xy = self.information_gain(x, y)
        h_x = self.entropy(x)
        h_y = self.entropy(y)

        if (h_x + h_y) != 0.0:
            return 2.0 * (ig_xy / (h_x + h_y))
        else:
            return 1.0

    # Get the next element of a given value
    def get_next_element(self, valid, fp, len_):

        # Get index of Fp
        fp_index = self.su_order.index(fp)

        # Search the next element
        for j in range(fp_index + 1, len_):
            if valid[j] == 1:
                return self.su_order[j]

        # No valid element found
        return -1

    # --------------- FCBF ALGORITHM ------------- #

    def execute(self):

        # Get the number of attributes (-1 to exclude class index)
        num_attributes = len(self.dataset[0]) - 1
        length = 0

        su_list_dup = [0] * num_attributes

        # Initializing su_list
        for i in range(num_attributes):

            # Compute SU
            su_value = self.su(i, num_attributes)

            su_list_dup[i] = su_value
            self.su_list.append(su_value)

            # Check if SU is higher than defined Delta
            if su_value > self.delta:
                length += 1

            # Set a list of all attributes to False with their SU values
            self.attributes_map.append([i, False, su_value])

        # Order attributes from max SU to min SU
        for i in range(length):

            max_ = 0.0
            max_index = -1

            for j in range(num_attributes):
                if su_list_dup[j] > max_:
                    max_ = su_list_dup[j]
                    max_index = j

            self.su_order.append(max_index)

            # Removing the max element in order to get the next maximum element in the next iteration
            su_list_dup[max_index] = 0.0

        # Initializing an array of 1 (All features are valid initially)
        valid = [1] * length

        # Get the feature with the highest SU value with respect to the class
        fp = self.su_order[0]

        # Iterate over all features with a SU value higher than delta till there aren't more Fp
        while fp != -1:

            # Get the next value of su
            fq = self.get_next_element(valid, fp, length)

            if fq != -1:

                while True:

                    fqd = fq

                    # Compute the SU of Fp and Fq
                    fp_fq = self.su(fp, fq)

                    # Compute the SU of Fq and Fc (Class)
                    fq_fc = self.su(fq, num_attributes)

                    # Check if Fp & Fq have a SU higher than Fq & Fc
                    # If YES remove Fq from the list ELSE keep it and go on
                    if fp_fq >= fq_fc:

                        # Get index of Fq and set it to 0
                        fq_index = self.su_order.index(fq)
                        valid[fq_index] = 0

                        # Get the next element of Fqd
                        fq = self.get_next_element(valid, fqd, length)

                    else:

                        # Get the next element of Fq
                        fq = self.get_next_element(valid, fq, length)

                    # If there aren't no more Fq exit from while
                    if fq == -1:
                        break

            # Get the next element of Fp
            fp = self.get_next_element(valid, fp, length)

        for i in range(length):
            if valid[i] != 0:
                # Set True to all valid attributes
                self.attributes_map[self.su_order[i]][1] = True

    # Get mapped attributes [index_attr, true/false, su_value]
    def get_mapped_attributes(self):

        # Sort attributes by su value from lower to higher
        self.attributes_map.sort(key=itemgetter(2))

        return self.attributes_map[::-1]
