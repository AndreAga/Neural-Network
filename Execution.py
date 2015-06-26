from Dataset import *
from NeuralNetwork import *
from VarianceThreshold import *
from FCBF import *
from operator import itemgetter

import sys
import os
import time
import math
import random
import matplotlib.pyplot as plot


class Output(object):

    # Write all to file and standard output
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)


class Initialization:

    def __init__(self, save_result, save_result_path, dataset_path, dataset_name, delimiter, num_attributes,
                 class_index, variance_min, variance_max, num_folds, num_shake, num_hidden_nodes_plus, num_epochs,
                 early_stop, learning_rate, momentum, val_percent, feature_type, fcbf_delta, fs_order, fs_criterion):

        # Set all variables
        self.delimiter = delimiter
        self.num_attributes = num_attributes
        self.class_index = class_index
        self.fs = feature_type
        self.variance_min = variance_min
        self.variance_max = variance_max
        self.fcbf_delta = fcbf_delta
        self.fs_order = fs_order
        self.fs_criterion = fs_criterion
        self.num_folds = num_folds
        self.val_percent = val_percent
        self.early_stop = early_stop
        self.num_shake = num_shake
        self.num_hidden_nodes_plus = num_hidden_nodes_plus
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.save_result = save_result
        self.save_result_path = save_result_path
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_uri = self.dataset_path + '/' + self.dataset_name + '.csv'

        # Start the execution
        self.start()

    # Set new filename
    def get_filename(self, extension):

        path = ''

        i = 0
        exist = True
        while exist:
            path = "{0}/{1}_{2}.{3}".format(self.save_result_path, self.dataset_name, i, extension)
            if not os.path.exists(path):
                exist = False
            i += 1

        return path

    def network_execution(self, dataset, num_input_nodes, mapped_classes,
                          neural_network, network_results):

        # Print number of considered attributes
        print("\nNeural Network: Execution with %i attributes" % num_input_nodes)

        # Variable to store all Cross-Validation results
        cross_validation_results = []

        # Get the size of a single subset
        subset_size = len(dataset)/self.num_folds

        # Start Cross-Validation
        for i in range(self.num_folds):

            # Create the new train and test datasets {[1,2,3,4,5,6,7,8,9,0] => [1,2,3,4,5,6,7] [8,9,0]}
            subset_train = dataset[:i * subset_size] + dataset[(i+1) * subset_size:]
            subset_test = dataset[i * subset_size:][:subset_size]

            # Get the size of the validation set
            validation_size = (len(subset_train) * self.val_percent) / 100

            # Create the new train and validation datasets {[1,2,3,4,5,6,7] => [1,2,3,4,5] [6,7]}
            if validation_size != 0:
                subset_train_new = subset_train[:-validation_size]
                subset_val = subset_train[-validation_size:]
            else:
                subset_train_new = subset_train
                subset_val = []

            # Train the Network
            print("\nTraining Network with the subset %i..." % (i+1))
            neural_network.train(subset_train_new, subset_val)
            print("Training Finished!")

            # Test the Network
            print("\nTest Network...")
            iter_result = neural_network.test(subset_test)

            # Put the result in the result list
            cross_validation_results.append(iter_result)
            print("Test Finished!")

            # Calculate the accuracy
            temp_accuracy = (float(iter_result[0]) / (float(iter_result[0]) + float(iter_result[1]))) * 100

            # Print the subset results
            print("\nResults with the subset %i:" % (i+1))
            print("\tRight:    {0}\n"
                  "\tWrong:    {1}\n"
                  "\tAccuracy: {2}%".format(iter_result[0], iter_result[1], round(temp_accuracy, 1)))

        correct = 0.0
        incorrect = 0.0

        # Get all results
        for result in cross_validation_results:
            correct += result[0]
            incorrect += result[1]

        # Compute the neural network result and store it
        accuracy = correct / (correct + incorrect)

        # Add new final result
        network_results.append([num_input_nodes, accuracy])

        # Print confusion matrix
        print("\nNeural Network: Performance")
        for row in neural_network.confusion_matrix:
            print(row)

        # Print mapped classes
        print('\nAttributes Legend:')
        print(str(mapped_classes))

        print("\nNeural Network: Final Accuracy\n"
              "\tWith {0} attributes: {1}%".format(num_input_nodes, round((accuracy * 100), 1)))

        return accuracy

    def start(self):

        # Start Time
        start_time = time.time()

        output_image_path = ''
        output_file = ''

        original_stdout = sys.stdout

        # Create the name of new output and image files
        if self.save_result:

            # Check if output path exist
            if not os.path.exists(self.save_result_path):
                os.makedirs(self.save_result_path)

            # Set output file name
            output_file_path = self.get_filename('txt')

            # Set output file name
            output_image_path = self.get_filename('png')

            # Copy console output to file
            output_file = open(output_file_path, 'w')
            sys.stdout = Output(sys.stdout, output_file)

        # Get the dataset from CSV file
        data = Dataset(self.dataset_uri, self.num_attributes, self.class_index, self.delimiter, self.fs)
        dataset_info = data.get_datasets()
        dataset = dataset_info[0]
        mapped_classes = data.get_mapped_class()

        # Define network variables
        num_input_nodes_original = dataset_info[1]
        num_input_nodes = dataset_info[1]
        num_hidden_nodes = dataset_info[1] + self.num_hidden_nodes_plus
        num_output_nodes = dataset_info[2]

        attributes_mapped = []

        # Execute Features Selection
        if (self.fs == "VT") or (self.fs == "FS"):

            # Initialize Variance Threshold
            vt = VarianceThreshold(dataset, num_input_nodes, self.variance_max, self.variance_min)
            vt.execute()

            # Get mapped attributes
            attributes_mapped = vt.get_mapped_attributes()

            # Reverse attributes order if features selection is Forward Selection {[5,4,3,2,1] => [1,2,3,4,5]}
            if self.fs == "FS":
                # Sort attributes by variance
                if self.fs_order == 1:
                    attributes_mapped.sort(key=itemgetter(2))
                if self.fs_order == 2:
                    attributes_mapped.sort(key=itemgetter(2))
                    attributes_mapped = attributes_mapped[::-1]
            elif self.fs == "VT":
                # Sort attributes by variance
                attributes_mapped.sort(key=itemgetter(2))
                attributes_mapped = attributes_mapped[::-1]

        elif self.fs == "FCBF":

            # Initialize FCBF
            fcbf = FCBF(dataset_info[3], self.fcbf_delta)
            fcbf.execute()

            #  Get mapped attributes
            attributes_mapped = fcbf.get_mapped_attributes()

        # Number between which the random weights will be generated
        rand_range = 1 / math.sqrt(num_input_nodes)

        # Create a new network
        neural_network = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes, self.num_epochs,
                                       self.early_stop, self.learning_rate, self.momentum, rand_range)

        # Shake the dataset many times {[1,2,3,4,5] => [2,1,5,3,4]} [THE ORDER IS RANDOM!]
        for shake in range(self.num_shake):
            random.shuffle(dataset)

        # Print some information
        print("\nCONFIGURATION\n")
        print("Dataset:            %s" % self.dataset_uri)
        print("Attributes Number:  %i" % (self.num_attributes - 1))
        print("Features Selection: %s" % self.fs)
        if self.fs == "VT":
            print("Variance Min:       %f" % self.variance_min)
            print("Variance Max:       %f" % self.variance_max)
        elif self.fs == "FCBF":
            print("Delta SU:           %s" % self.fcbf_delta)
        elif self.fs == "FS":
            if self.fs_order == 0:
                print("Features Order:     %s" % 'Original')
            elif self.fs_order == 1:
                print("Features Order:     %s" % 'Low -> High')
            else:
                print("Features Order:     %s" % 'High -> Low')
            if self.fs_criterion == 0:
                print("Condition:          %s" % '>=')
            else:
                print("Condition:          %s" % '>')
        print("CrossValidation:    %i folds" % self.num_folds)
        print("Epochs:             %i" % self.num_epochs)
        print("Learning Rate:      %f" % self.learning_rate)
        print("Momentum:           %f" % self.momentum)
        if self.val_percent != 0:
            print("Validation Set:     %i%%" % self.val_percent)
            print("Early Stop:         %f" % self.early_stop)
        print("\nFeatures Selection: Attributes Relevance")
        for attribute in attributes_mapped:
            print("Attribute %i:" % (attribute[0] + 1))
            if self.fs == "VT" or self.fs == "FCBF":
                print("\tRelevant: {0}".format(attribute[1]))
            if self.fs == "VT" or self.fs == "FS":
                print("\tVariance: {0}".format(round(attribute[2], 3)))
            if self.fs == "FCBF":
                print("\tSU Value: {0}".format(round(attribute[2], 3)))

        # Define two nested lists
        dataset_attributes = [[0] * num_input_nodes] * len(dataset)
        dataset_targets = [[0] * num_output_nodes] * len(dataset)

        # Divide the dataset in attributes and targets
        i = 0
        for row in dataset:
            dataset_attributes[i] = row[:num_input_nodes]
            dataset_targets[i] = row[-num_output_nodes:]
            i += 1

        # Set some support variables
        dataset_attributes_transposed = map(list, zip(*dataset_attributes))
        new_dataset_edit = dataset_attributes_transposed[:]
        new_dataset_support = dataset_attributes_transposed[:]
        new_dataset_fs = []

        # Store the accuracies of network
        network_results = []

        # Store best attributes
        final_attr_list = []

        it = 0
        skip = False
        attributes_mapped.append([[], [], []])
        max_accuracy = 0

        if (self.fs == "VT") or (self.fs == "FCBF"):

            # Iterate over the list of mapped attributes to remove an attribute at a time
            for attr in attributes_mapped:

                # ------------ NETWORK EXECUTION  ------------ #

                if not skip:

                    self.network_execution(dataset, num_input_nodes, mapped_classes,
                                           neural_network, network_results)

                # ------ FEATURES SELECTIONS EXECUTION  ------ #

                if it < len(attributes_mapped)-1:

                    # Extract Attribute Values
                    index = attr[0]
                    value = attr[1]
                    var = attr[2]

                    # Check if the attribute is True or False
                    if not value:

                        # Skip if is the last attribute negative
                        if len(new_dataset_edit) > 1:

                            # Print Feature Selection information
                            print("\nFeature Selection: Attribute Elimination")
                            print("\tIndex:    %i" % (index + 1))
                            if self.fs == "VT":
                                print("\tVariance: %.3f" % var)
                            elif self.fs == "FCBF":
                                print("\tSU Value: %.3f" % var)

                            # Remove from dataset the row corresponding to index
                            new_dataset_edit.remove(new_dataset_support[index])

                            dataset_attributes_transposed = map(list, zip(*new_dataset_edit))

                            # Define new network variables
                            num_input_nodes = len(dataset_attributes_transposed[0])
                            num_hidden_nodes = len(dataset_attributes_transposed[0]) + self.num_hidden_nodes_plus

                            # Reconstruct the dataset [rows + targets]
                            dataset = []
                            z = 0
                            for row in dataset_attributes_transposed:
                                dataset.append(row + dataset_targets[z])
                                z += 1

                            # Define a new network
                            neural_network = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes,
                                                           self.num_epochs, self.early_stop, self.learning_rate,
                                                           self.momentum, rand_range)
                            skip = False
                        else:
                            skip = True
                    else:
                        skip = True

                it += 1

        elif self.fs == "FS":

            # Iterate over the list of mapped attributes to remove an attribute at a time
            for attr in attributes_mapped:

                # ------------ NETWORK EXECUTION  ------------ #

                if not skip and it != 0:

                    tmp_accuracy = self.network_execution(dataset, num_input_nodes, mapped_classes,
                                                          neural_network, network_results)

                # ------ FEATURES SELECTIONS EXECUTION  ------ #

                    print '\nEvaluated Features: ', final_attr_list

                    # If it's better update it, otherwise remove the feature
                    if self.fs_criterion == 1:
                        if tmp_accuracy > max_accuracy:
                            print "\nNew Best Accuracy: {0}%".format(round((tmp_accuracy * 100), 1))
                            max_accuracy = tmp_accuracy
                        else:
                            if len(final_attr_list) > 0:
                                final_attr_list.pop()
                            if len(new_dataset_fs) > 0:
                                new_dataset_fs.pop()
                            if len(network_results) > 0:
                                network_results.pop()
                    else:
                        if tmp_accuracy >= max_accuracy:
                            print "\nNew Best Accuracy: {0}%".format(round((tmp_accuracy * 100), 1))
                            max_accuracy = tmp_accuracy
                        else:
                            if len(final_attr_list) > 0:
                                final_attr_list.pop()
                            if len(new_dataset_fs) > 0:
                                new_dataset_fs.pop()
                            if len(network_results) > 0:
                                network_results.pop()

                if it < len(attributes_mapped)-1:

                    # Extract Attribute Values
                    index = attr[0]
                    var = attr[2]

                    # Check if the feature variance is not 0, otherwise skip it
                    if var != 0:

                        # Print Feature Selection information
                        print("\nFeature Selection: Adding Attribute")
                        print("\tIndex:   %i" % (index + 1))

                        # Insert next feature
                        final_attr_list.append(index + 1)
                        new_dataset_fs.append(new_dataset_support[index])
                        dataset_attributes_transposed = map(list, zip(*new_dataset_fs))

                        # Define new network nodes
                        num_input_nodes = len(dataset_attributes_transposed[0])
                        num_hidden_nodes = len(dataset_attributes_transposed[0]) + self.num_hidden_nodes_plus

                        # Reconstruct the dataset [rows + targets]
                        dataset = []
                        z = 0
                        for row in dataset_attributes_transposed:
                            dataset.append(row + dataset_targets[z])
                            z += 1

                        # Define a new network
                        neural_network = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes,
                                                       self.num_epochs, self.early_stop, self.learning_rate,
                                                       self.momentum, rand_range)
                        skip = False

                    else:
                        print("\nFeature Selection: Skip Attribute")
                        print("\tIndex:    %i" % (index + 1))
                        print("\tVariance: 0.0")

                        skip = True

                it += 1

        else:

            # ------------ NETWORK EXECUTION  ------------ #

            self.network_execution(dataset, num_input_nodes, mapped_classes,
                                   neural_network, network_results)

        if self.fs == "FS":
            # Print the list of best features
            print "\nFinal Best Attributes: ", final_attr_list

        # Get final accuracies and number of attributes
        attributes = []
        accuracies = []
        for result in network_results:
            attributes.append(result[0])
            accuracies.append(result[1])

        # Get the final number of attributes
        if self.fs == "FS":
            len_attributes = len(final_attr_list)
        elif (self.fs == "VT") or (self.fs == "FCBF"):
            len_attributes = num_input_nodes_original
        else:
            len_attributes = num_input_nodes

        # Create a plot to show the final network results
        fig = plot.figure()
        fig.canvas.set_window_title(self.dataset_name)
        plot.title('Neural Network: Final Results {0}\n'.format(self.dataset_name))
        plot.xlabel('Attributes')
        plot.ylabel('Accuracy')
        plot.grid(True)
        plot.axis([0, len_attributes+1, 0, 1])
        plot.plot(attributes, accuracies, 'ro')

        end_time = time.time()
        print '\nTime Elapsed: ', int((end_time - start_time) / 60.0), ' minutes'

        print("\nExecution Ended!")

        # Show Results
        fig.show()

        sys.stdout = original_stdout

        # Write all results
        if self.save_result:
            plot.savefig(output_image_path)
            output_file.close()

        # -------------------------------------------- #
