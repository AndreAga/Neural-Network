from compiler.ast import flatten
import csv
import copy


class Dataset:
    def __init__(self, dataset_name, num_attributes, class_index, csv_delimiter, features_selection):

        # Initialize variables
        self.dataset_name = dataset_name
        self.num_attributes = num_attributes
        self.class_index = class_index - 1
        self.delimiter = csv_delimiter
        self.features_selection = features_selection

        # Create support variables
        self.dataset = []
        self.dataset_new_nn = []
        self.dataset_new_fcbf = []
        self.attributes_unique = []
        self.attributes_new = []
        self.attributes_support = []
        self.classes = []
        self.num_output = 0

    def create_dataset(self, targets):

        dataset_copy = copy.deepcopy(self.dataset)
        dataset_final = []

        # Iterate over all unique attributes
        for i in range(len(self.attributes_unique)):

            # Extract a set of unique attributes
            extracted_attr = self.attributes_unique[i]

            # Iterate over all new attributes values
            for j in range(len(targets[i])):

                # Extract an attribute from previous set
                class_popped = list(extracted_attr)[j]

                # Iterate over dataset
                z = 0
                for row in self.dataset:
                    # Check if attributes are not empty
                    if row[i]:
                        # If the attribute of row is equal to the extracted attribute
                        if row[i] == class_popped:
                            # Put in the new dataset the new attribute values
                            dataset_copy[z][i] = targets[i][j]
                    z += 1

        # Create the final dataset for Feature Selection
        z = 0
        for row in dataset_copy:

            # List flatten {[[1], [2], [3]] => [1, 2, 3]}
            flatten_row = flatten(row)

            new_row = []
            for attr in range(len(flatten_row)):
                # Make all attribute values float
                new_row.append(float(flatten_row[attr]))

            dataset_final.append(new_row)
            z += 1

        return dataset_final

    def get_datasets(self):

        # Read dataset and add rows in dataset
        with open(self.dataset_name, 'rb') as csv_dataset:

            # Initialization of csv reader
            reader = csv.reader(csv_dataset, delimiter=self.delimiter)

            # Iterate over all csv rows
            for row in reader:

                # Get all attributes
                self.classes.append(row[self.class_index])

                # Get the left and right list without target
                left = row[:self.class_index]
                right = row[self.class_index + 1:]

                # Put the target at the end of the concatenated list and add the new row to dataset
                self.dataset.append(left + right + [row[self.class_index]])
                self.attributes_support.append(row[self.class_index])

        # Get all attributes
        for attr in range(self.num_attributes):

            at = []
            for row in self.dataset:

                # Check if attr index is equal to class index (class has to be changed)
                if attr == self.num_attributes - 1:
                    at.append(row[attr])
                else:

                    # Check if attribute is in the format: ,.N
                    if row[attr][0] == '.':
                        new_attr = '0' + row[attr]
                        point = True
                    else:
                        point = False

                    if point:
                        # Check if the modified attribute is not a number
                        if not self.is_number(new_attr):
                            at.append(new_attr)
                    else:
                        # Check if the attribute is not a number
                        if not self.is_number(row[attr]):
                            at.append(row[attr])

            # Add this attribute to the list
            self.attributes_unique.append(set(at))

        list_target = []

        z = 0
        for attribute in self.attributes_unique:
            # Create an identity matrix with class dimension for target values
            if z == self.num_attributes - 1:
                self.attributes_new.append([[1 if i == j else 0 for j in range(len(attribute))]
                                            for i in range(len(attribute))])
                self.num_output = len(attribute)
            # Create a list with incremental values for other attributes
            else:
                self.attributes_new.append([[i + 1] for i in range(len(attribute))])
            list_target.append([[i + 1] for i in range(len(attribute))])
            z += 1

        # Create the final dataset for Neural Network
        self.dataset_new_nn = self.create_dataset(self.attributes_new)

        # Create the final dataset for FCBF
        if self.features_selection == "FCBF":
            self.dataset_new_fcbf = self.create_dataset(list_target)

        return [self.dataset_new_nn, self.num_attributes - 1, self.num_output, self.dataset_new_fcbf]

    # Create the dictionary for classes
    def get_mapped_class(self):

        # Define the dictionary
        classes_mapped = {}

        # Get single classes
        classes_temp = set(self.attributes_support)

        length = len(classes_temp)
        for i in range(length):
            # Add the {key: value} to the dictionary
            classes_mapped.update({(i + 1): classes_temp.pop()})

        return classes_mapped

    # Check if a string is numeric
    @staticmethod
    def is_number(x):

        try:
            float(x)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(x)
            return True
        except (TypeError, ValueError):
            pass

        return False
