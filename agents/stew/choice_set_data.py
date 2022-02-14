import numpy as np


class ChoiceSetData:
    """
    Data structure to store choice sets. See Table 1 in the Supplementary Material of the article.
    """

    def __init__(self, num_features, max_choice_set_size, max_number_of_choice_sets=np.inf):
        self.num_features = num_features
        self.data = np.zeros((0, self.num_features + 2))
        self.choice_set_counter = 0.
        self.current_number_of_choice_sets = 0.
        self.max_choice_set_size = max_choice_set_size
        self.max_number_of_choice_sets = max_number_of_choice_sets

    def push(self, features, choice_index, delete_oldest=False):
        choice_set_len = len(features)
        one_hot_choice = np.zeros((choice_set_len, 1))
        one_hot_choice[choice_index] = 1.
        choice_set_index = np.full(shape=(choice_set_len, 1), fill_value=self.choice_set_counter)
        self.data = np.vstack((self.data, np.hstack((choice_set_index, one_hot_choice, features))))
        self.choice_set_counter += 1.
        self.current_number_of_choice_sets += 1.
        if delete_oldest or self.current_number_of_choice_sets > self.max_number_of_choice_sets:
            first_choice_set_index = self.data[0, 0]
            for ix in range(self.max_choice_set_size+1):
                if self.data[ix, 0] != first_choice_set_index:
                    break
            if ix > self.max_choice_set_size:
                raise ValueError("Choice set should not be higher than " + str(self.max_choice_set_size))
            self.data = self.data[ix:]
            if self.current_number_of_choice_sets > 0:
                self.current_number_of_choice_sets -= 1.

    def sample(self):
        # Currently just returns a copy of the entire data set.
        return self.data.copy()

    def delete_data(self):
        self.data = np.zeros((0, self.num_features + 2))
        self.current_number_of_choice_sets = 0
        self.choice_set_counter = 0










