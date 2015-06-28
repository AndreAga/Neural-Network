# -------------------------------------------- #
# --------- NEURAL NETWORK DESCRIPTION ------- #
# -------------------------------------------- #
#                                              #
#   Type Problem:  Classification              #
#   Type Network:  Multi-Layer                 #
#   Type Target:   Multi-Target                #
#                                              #
# -------------------------------------------- #
#                                              #
#   Hidden Activation:  Hyperbolic Tangent     #
#   Output Activation:  SoftMax                #
#                                              #
# -------------------------------------------- #

from __future__ import print_function
import random
import math


class NeuralNetwork:

    # ---------- NETWORK INITIALIZATION  ---------- #

    def __init__(self, num_input, num_hidden, num_output, num_epochs, early_stop, learning_rate, momentum, rand_range):

        # Set when to stop leaning
        self.early_stop = early_stop

        # Set the number of nodes
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        # Set training epochs
        self.num_epochs = num_epochs

        # Set backpropagation variables
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Set the upper and lower random bounds
        self.upper_bound = rand_range
        self.lower_bound = -rand_range

        # Set lists of zeros to input and output nodes
        self.inputs = self.generate_matrix(False, 1, self.num_input)[0]
        self.outputs = self.generate_matrix(False, 1, self.num_output)[0]

        # Set the confusion matrix
        self.confusion_matrix = self.generate_matrix(False, self.num_output, self.num_output)

        # Set a matrix of random values (Weight of Edges)
        self.weights_inputs_hidden = self.generate_matrix(True, self.num_input, self.num_hidden)
        self.weights_hidden_hidden = self.generate_matrix(True, self.num_hidden, self.num_hidden)
        self.weights_hidden_outputs = self.generate_matrix(True, self.num_hidden, self.num_output)

        # Set the biases for hidden and output nodes with random values
        self.hidden_biases_1 = self.generate_matrix(True, 1, self.num_hidden)[0]
        self.hidden_biases_2 = self.generate_matrix(True, 1, self.num_hidden)[0]
        self.output_biases = self.generate_matrix(True, 1, self.num_output)[0]

        # Set a support list for the hidden gradients
        self.hidden_outputs = self.generate_matrix(False, 1, self.num_hidden)[0]
        self.hidden_hidden = self.generate_matrix(False, 1, self.num_hidden)[0]

    # -------------------------------------------- #

    # -------------- TRAIN NETWORK  -------------- #

    def train(self, train_set, validation_set):

        # Define two errors (arbitrary values)
        old_error = 100001
        new_error = 100000

        # Execute the train steps many times
        for epoch in range(self.num_epochs):

            # Iterate over the set elements
            for train_i in range(len(train_set)):

                # Get attribute and target values
                values = self.divide_tuple(train_set, train_i)

                # Compute the classification given the attribute values
                self.evaluate_classification(values[0])

                # Execute Backpropagation given the target values
                self.backpropagation(values[1])

            # Check if validation set is defined then validate the network
            if len(validation_set) > 0:

                tmp_error = old_error
                old_error = new_error

                # Compute the partial accuracy of network
                error_sum = 0
                for val_i in range(len(validation_set)):

                    # Get attribute and target values
                    values = self.divide_tuple(validation_set, val_i)

                    # Compute Output on the attribute values
                    self.evaluate_classification(values[0])

                    # Get the sum of output errors
                    for i in range(len(values[1])):
                        error_sum += (values[1][i] - self.outputs[i]) ** 2

                # Compute Error = SUM[(Xt - Xn)^2] / 2
                new_error = error_sum / len(validation_set)

                # If minimum error has been reached stop
                # It uses two errors to ensure that small fluctuations don't change early stopping in premature stopping
                if ((old_error - new_error) <= self.early_stop) or ((tmp_error - old_error) <= self.early_stop):
                    break

    # -------------------------------------------- #

    # --------------- TEST NETWORK  -------------- #

    def test(self, test_set):

        # Number of tests classified correctly
        corrects = 0

        # Iterate over the set elements
        for test_i in range(len(test_set)):

            # Get attribute and target values
            values = self.divide_tuple(test_set, test_i)

            # Compute Output on the attribute values
            self.evaluate_classification(values[0])

            # Get the index of the MAX value in the output
            index = self.outputs.index(max(self.outputs))

            # Get the index of the real class
            real_max = values[1].index(max(values[1]))

            # Update the confusion matrix
            self.confusion_matrix[real_max][index] += 1

            # Check if the index of the classification is the same of the target
            if real_max == index:
                corrects += 1

        return [corrects, (len(test_set) - corrects)]

    # -------------------------------------------- #

    # ---------- NETWORK CLASSIFICATION  --------- #

    def evaluate_classification(self, attribute_values):

        hidden_sums_1 = self.generate_matrix(False, 1, self.num_hidden)[0]
        hidden_sums_2 = self.generate_matrix(False, 1, self.num_hidden)[0]
        output_sums = self.generate_matrix(False, 1, self.num_output)[0]

        ############################################
        # STEP 0: Initialization of Input nodes    #
        ############################################
        # Copy attribute values to input nodes.    #
        ############################################

        self.inputs = attribute_values[:]

        ############################################
        # STEP 1: Go from Input to Hidden1  nodes  #
        ############################################
        # 1A. Initialize hidden_sums with the sum  #
        # of input value multiplied by his weight. #
        # 1B. Sum the previous number by the bias  #
        # of the respective hidden node.           #
        # 1C. Active the hidden node.              #
        ############################################

        for eo_i in range(self.num_hidden):
            # Step 1A
            for eo_j in range(self.num_input):
                hidden_sums_1[eo_i] += (self.inputs[eo_j] * self.weights_inputs_hidden[eo_j][eo_i])
            # Step 1B
            hidden_sums_1[eo_i] += self.hidden_biases_1[eo_i]
            # Step 1C
            self.hidden_hidden[eo_i] = self.hidden_activation(hidden_sums_1[eo_i])

        ############################################
        # STEP 2: Go from Hidden1 to Hidden2 nodes #
        ############################################
        # 1A. Initialize hidden_sums with the sum  #
        # of input value multiplied by his weight. #
        # 1B. Sum the previous number by the bias  #
        # of the respective hidden node.           #
        # 1C. Active the hidden node.              #
        ############################################

        for eo_i in range(self.num_hidden):
            # Step 1A
            for eo_j in range(self.num_hidden):
                hidden_sums_2[eo_i] += (self.hidden_hidden[eo_j] * self.weights_hidden_hidden[eo_j][eo_i])
            # Step 1B
            hidden_sums_2[eo_i] += self.hidden_biases_2[eo_i]
            # Step 1C
            self.hidden_outputs[eo_i] = self.hidden_activation(hidden_sums_2[eo_i])

        ############################################
        # STEP 3: Go from Hidden1 to Output nodes  #
        ############################################
        # 1A. Initialize output_sums with the sum  #
        # of hidden value multiplied by his weight.#
        # 1B. Sum the previous number by the bias  #
        # of the respective output node.           #
        # 1C. Active the output node.              #
        ############################################

        for eo_i in range(self.num_output):
            # Step 2A
            for eo_j in range(self.num_hidden):
                output_sums[eo_i] += (self.hidden_outputs[eo_j] * self.weights_hidden_outputs[eo_j][eo_i])
            # Step 2B
            output_sums[eo_i] += self.output_biases[eo_i]
        # Step 2C
        self.outputs = self.output_activation(output_sums)

    # -------------------------------------------- #

    # ------------- BACKPROPAGATION  ------------- #

    def backpropagation(self, target):

        #############################################
        # STEP 0: Initialization of required Lists  #
        #############################################

        # Set the gradient lists
        hidden_gradients_1 = self.generate_matrix(False, 1, self.num_hidden)[0]
        hidden_gradients_2 = self.generate_matrix(False, 1, self.num_hidden)[0]
        output_gradients = self.generate_matrix(False, 1, self.num_output)[0]

        # Set the lists to store delta values. This deltas will be used with momentum
        input_hidden_deltas = self.generate_matrix(False, self.num_input, self.num_hidden)
        hidden_hidden_deltas = self.generate_matrix(False, self.num_hidden, self.num_hidden)
        hidden_output_deltas = self.generate_matrix(False, self.num_hidden, self.num_output)

        # Set the lists to store bias values. This biases will be used with momentum
        hidden_bias_deltas_1 = self.generate_matrix(False, 1, self.num_hidden)[0]
        hidden_bias_deltas_2 = self.generate_matrix(False, 1, self.num_hidden)[0]
        output_bias_deltas = self.generate_matrix(False, 1, self.num_output)[0]

        #############################################
        # STEP 1: Compute Gradients                 #
        #############################################
        # Gradients are values that measure of how  #
        # far off, and in what direction (positive  #
        # or negative), the current neural network  #
        # output values are, compared to the target #
        #############################################
        # The gradient must be calculated from the  #
        # right to the left, because the gradients  #
        # of the hidden nodes depend on the values  #
        # of the gradients of output nodes.         #
        #############################################

        # Output Gradients: output_activation_derivative * (desired_target - computed_output)
        for bp_i in range(self.num_output):
            # Calculate the error: difference between target and output
            error = target[bp_i] - self.outputs[bp_i]
            # Calculate gradient
            output_gradients[bp_i] = self.output_activation_derivative(self.outputs[bp_i]) * error

        # Hidden Gradients: hidden_activation_derivative * sum(G * W)
        # G = All output gradients
        # W = Weights from Hidden_2 to Output
        for bp_i in range(self.num_hidden):
            # Calculate the error: the sum of output gradients * hidden to output weights
            error = 0
            for bp_j in range(self.num_output):
                error += output_gradients[bp_j] * self.weights_hidden_outputs[bp_i][bp_j]
            # Calculate gradient
            hidden_gradients_2[bp_i] = self.hidden_activation_derivative(self.hidden_outputs[bp_i]) * error

        # Hidden Gradients: hidden_activation_derivative * sum(G * W)
        # G = All hidden gradients
        # W = Weights from Hidden_1 to Hidden_2
        for bp_i in range(self.num_hidden):
            # Calculate the error: the sum of hidden gradients * hidden to hidden weights
            error = 0
            for bp_j in range(self.num_hidden):
                error += hidden_gradients_2[bp_j] * self.weights_hidden_hidden[bp_i][bp_j]
            # Calculate gradient
            hidden_gradients_1[bp_i] = self.hidden_activation_derivative(self.hidden_hidden[bp_i]) * error

        #############################################
        # STEP 2: Update Weights                    #
        #############################################
        # Weight Formula: Delta + (PWD * Momentum)  #
        #############################################
        # Delta = GTN * OFN * LR                    #
        # GTN = Gradient To-Node                    #
        # OFN = Output From-Node                    #
        # LR = Learning Rate                        #
        # PWD = Previous Weight Delta               #
        #############################################

        # Hidden_2 -> Output
        for bp_i in range(self.num_hidden):
            for bp_j in range(self.num_output):
                # Calculate the delta
                delta_value = output_gradients[bp_j] * self.hidden_outputs[bp_i] * self.learning_rate
                # Calculate the increment of the new weight
                increment = hidden_output_deltas[bp_i][bp_j] * self.momentum
                # Calculate the new weight
                self.weights_hidden_outputs[bp_i][bp_j] += delta_value + increment
                # Store delta for the next iteration
                hidden_output_deltas[bp_i][bp_j] = delta_value

        # Hidden_1 -> Hidden_2
        for bp_i in range(self.num_hidden):
            for bp_j in range(self.num_hidden):
                # Calculate the delta
                delta_value = hidden_gradients_2[bp_j] * self.hidden_hidden[bp_i] * self.learning_rate
                # Calculate the increment of the new weight
                increment = hidden_hidden_deltas[bp_i][bp_j] * self.momentum
                # Calculate the new weight
                self.weights_hidden_hidden[bp_i][bp_j] += delta_value + increment
                # Store delta for the next iteration
                hidden_hidden_deltas[bp_i][bp_j] = delta_value

        # Input -> Hidden_1
        for bp_i in range(self.num_input):
            for bp_j in range(self.num_hidden):
                # Calculate the delta
                delta_value = hidden_gradients_1[bp_j] * self.inputs[bp_i] * self.learning_rate
                # Calculate the increment of the new weight
                increment = input_hidden_deltas[bp_i][bp_j] * self.momentum
                # Calculate the new weight
                self.weights_inputs_hidden[bp_i][bp_j] += delta_value + increment
                # Store delta for the next iteration
                input_hidden_deltas[bp_i][bp_j] = delta_value

        #############################################
        # STEP 3: Update Biases                     #
        #############################################
        # Bias Formula: Delta + (PWD * Momentum)    #
        #############################################
        # Delta = CG * LR                           #
        # CG = Gradient of Current Node             #
        # LR = Learning Rate                        #
        # PWD = Previous Weight Delta               #
        #############################################

        # Output Biases
        for bp_i in range(self.num_output):
            # Calculate the delta
            delta_value = output_gradients[bp_i] * self.learning_rate
            # Calculate the new bias
            self.output_biases[bp_i] += delta_value + (output_bias_deltas[bp_i] * self.momentum)
            # Store delta for the next iteration
            output_bias_deltas[bp_i] = delta_value

        # Hidden_2 Biases
        for bp_i in range(self.num_hidden):
            # Calculate the delta
            delta_value = hidden_gradients_2[bp_i] * self.learning_rate
            # Calculate the new bias
            self.hidden_biases_2[bp_i] += delta_value + (hidden_bias_deltas_2[bp_i] * self.momentum)
            # Store delta for the next iteration
            hidden_bias_deltas_2[bp_i] = delta_value

        # Hidden_1 Biases
        for bp_i in range(self.num_hidden):
            # Calculate the delta
            delta_value = hidden_gradients_1[bp_i] * self.learning_rate
            # Calculate the new bias
            self.hidden_biases_1[bp_i] += delta_value + (hidden_bias_deltas_1[bp_i] * self.momentum)
            # Store delta for the next iteration
            hidden_bias_deltas_1[bp_i] = delta_value

    # -------------------------------------------- #

    # ---------- ACTIVATION FUNCTIONS  ----------- #

    ################################################
    #             HYPERBOLIC TANGENT               #
    ################################################
    # Formula:                                     #
    #              (1 - e^(-2 * x))                #
    #      TANH =  ----------------                #
    #              (1 - e^(-2 * x))                #
    #                                              #
    ################################################
    # Derivative:                                  #
    #                                              #
    #    TANH' =  (1 - TANH(x)) * (1 + TANH(x))    #
    #                                              #
    ################################################
    #           HIDDEN ACTIVATION NODES            #
    ################################################

    @staticmethod
    def hidden_activation(x):
        if x > 20:
            return 1.0
        elif x < -20:
            return -1.0
        else:
            return (1 - math.exp(-2 * x)) / (1 + math.exp(-2 * x))

    @staticmethod
    def hidden_activation_derivative(x):
        return (1 - x) * (1 + x)

    ################################################
    #  SOFTMAX FUNCTION (NORMALIZED EXPONENTIAL)   #
    ################################################
    # It normalizes the values of a list in order  #
    # to reduce the influence of extreme values    #
    # or outliers in the data without removing     #
    # them from the dataset.                       #
    ################################################
    # Formula:                                     #
    #                  e^(x[i])                    #
    #         SM = ---------------                 #
    #               sum(e^(x[i]))                  #
    #                                              #
    ################################################
    # Derivative:                                  #
    #                                              #
    #          SM' =  (1 - SM(x)) * SM(x)          #
    #                                              #
    ################################################
    #           OUTPUT ACTIVATION NODES            #
    ################################################

    @staticmethod
    def output_activation(x):

        normalized = [0] * len(x)
        summation = 0

        # Compute the sum
        for sm_i in range(len(x)):
            summation += math.exp(x[sm_i])

        # Normalize all values
        for sm_j in range(len(x)):
            normalized[sm_j] = math.exp(x[sm_j]) / summation

        return normalized

    @staticmethod
    def output_activation_derivative(x):
        return (1 - x) * x

    # -------------------------------------------- #

    # ------------ HELPER FUNCTIONS  ------------- #

    # Generate a random number between lower and upper bound
    def generate_rand(self):
        return ((self.lower_bound - self.upper_bound) * random.random()) + self.upper_bound

    # Generate a matrix given the numbers of rows and columns and fill it with 0 or random number
    def generate_matrix(self, rand, rows, columns):

        matrix = []
        for i in range(rows):

            row = []
            for j in range(columns):

                if rand:
                    row.append(self.generate_rand())
                else:
                    row.append(0)

            matrix.append(row)

        return matrix

    # Divide the tuples of dataset in attribute and target lists
    def divide_tuple(self, data, index):

        attribute_values = [0] * self.num_input
        target_values = [0] * self.num_output

        # Get attributes elements {[1,2,3,4,5] => A=[1,2,3]}
        for data_j in range(self.num_input):
            attribute_values[data_j] = data[index][data_j]

        # Get target elements {[1,2,3,4,5] => T=[4,5]}
        for data_j in range(self.num_output):
            target_values[data_j] = data[index][data_j + self.num_input]

        return [attribute_values, target_values]

    # -------------------------------------------- #
