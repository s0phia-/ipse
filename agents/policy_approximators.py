import numpy as np
from scipy.stats import binom_test

from agents.stew.choice_set_data import ChoiceSetData
from agents.stew.mlogit import StewMultinomialLogit

import warnings


class PolicyApproximator:
    """
    Parent/base class from which other policy approximators inherit.
    This class provides the choose_action() method.
    The placeholder methods append_data() and fit() have to be implemented in the child classes!
    """
    def __init__(
        self,
        num_features,
        max_choice_set_size,
        max_number_of_choice_sets=np.inf,
        policy_weights=None,
        verbose=False
    ):
        # Init data handler
        self.num_features = num_features
        self.max_choice_set_size = max_choice_set_size
        self.max_number_of_choice_sets = max_number_of_choice_sets
        self.data_set = ChoiceSetData(num_features=self.num_features,
                                      max_choice_set_size=self.max_choice_set_size,
                                      max_number_of_choice_sets=self.max_number_of_choice_sets)

        if policy_weights is None:
            self.policy_weights = np.random.normal(loc=0, scale=1, size=self.num_features)
        else:
            self.policy_weights = policy_weights

        self.verbose = verbose

    def append_data(self, **rollout):
        """
        Processes rollout data (e.g., from BatchRollout.perform_rollouts() ) and adds the
        processed data to self.data_set.
        """
        pass

    def fit(self):
        """
        Samples a new data set from self.data_set and learns a new self.policy_weights.
        """
        pass

    def choose_action(self, choice_set, stochastic=False):
        """
        Compute utilities of all actions (dot product between action's feature vector and self.policy_weights)
        and choose action:
            - randomly among utility-maximizing action (when stochastic=False), or
            - via softmax probabilities (when stochastic=True).

        :param choice_set: a Numpy array of dimensions (num_actions, num_features)
        :param stochastic: a boolean indicating whether the policy is stochastic or not (see description above)
        :return: selected_action_index: an integer
        """
        action_utilities = choice_set.dot(self.policy_weights)
        if stochastic:
            choice_probabilities = self.softmax(action_utilities)
            selected_action_index = np.random.choice(len(choice_set), p=choice_probabilities)
        else:
            max_indices = np.where(action_utilities == np.max(action_utilities))[0]
            selected_action_index = np.random.choice(max_indices)
        return selected_action_index

    def v_print(self, a_string):
        """
        Prints conditional on the self.verbose switch. This saves a lot of lines of

        if self.verbose:
            print("blabla")

        :param a_string: string to print
        """
        if self.verbose:
            print(a_string)

    @staticmethod
    def softmax(utilities):
        """
        Calculates softmax probabilities for a NumPy array of "utilities" (or "scores")
        Uses the "max-trick" to avoid numerical instabilities.

        :param utilities: a NumPy array of (unnormalized) utilities or scores
        :return: a NumPy array of softmax probabilities
        """
        probabilities = np.exp(utilities - np.max(utilities))
        probabilities /= np.sum(probabilities)
        return probabilities


class MultinomialLogisticRegression(PolicyApproximator):
    """
    Approximates the policy using a (linear) multinomial logistic regression model.

    Supports different regularization terms and methods (cross-validation or fixed lambda).
    """
    def __init__(
            self,
            num_features,
            max_choice_set_size,
            max_number_of_choice_sets=np.inf,
            regularization=False,
            regularization_method="stew",
            regularization_strength=0.0,
            regularization_strength_method="fixed",
            policy_weights=None,
            verbose=False
    ):
        super().__init__(
            num_features=num_features,
            max_choice_set_size=max_choice_set_size,
            max_number_of_choice_sets=max_number_of_choice_sets,
            policy_weights=policy_weights,
            verbose=verbose
        )

        # Init (regularized) multinomial logistic regression model
        self.regularization = regularization
        self.regularization_strength = regularization_strength
        if self.regularization:
            self.regularization_method = regularization_method
            self.regularization_strength_method = regularization_strength_method
            assert self.regularization_method in ["stew", "ridge"], \
                "regularization_method has to be 'ridge' or 'stew'."
            assert self.regularization_strength_method in ["fixed", "cv"], \
                "regularization_strength_method has to be 'fixed' or 'cv' (=cross-validation)."
            if self.regularization_strength_method == "fixed":
                assert self.regularization_strength is not None and self.regularization_strength >= 0.0, \
                    "If regularization_strength_method == 'fixed' you have to provide a regularization_strength >= 0.0"
            elif self.regularization_strength_method == "cv" and self.regularization_strength is not None:
                warnings.warn("You specified regularization_strength_method == 'cv' but also provided a "
                              "regularization_strength. The given regularization_strength will be ignored!")
            if self.regularization_method == "stew":
                D = self.create_stew_matrix(self.num_features)
            elif self.regularization_method == "ridge":
                D = self.create_ridge_matrix(self.num_features)
        else:
            assert self.regularization_strength == 0.0, \
                "If regularization is turned off, you have to use regularization_strength == 0.0," \
                "which is also the default value!"
            D = np.zeros((self.num_features, self.num_features))

        self.model = StewMultinomialLogit(num_features=self.num_features, D=D)

    def append_data(self, **rollout):
        """
        Appends rollout data to the self.data_set.

        :param rollout: a dictionary containing
            - state_action_features, a 3D NumPy array of floats of shape (self.rollout_set_size, self.num_actions, self.num_features)
            - state_action_values, a 2D NumPy array of floats of shape (self.rollout_set_size, self.num_actions)
            - num_available_actions, a 1D NumPy array of floats of size self.rollout_set_size
            - did_rollout, a 1D NumPy array of floats of size self.rollout_set_size
        """
        for ix in range(len(rollout['state_action_values'])):
            if rollout['did_rollout'][ix]:
                num_available_actions_ix = rollout['num_available_actions'][ix]
                action_features = rollout['state_action_features'][ix][:num_available_actions_ix, :]
                action_values = rollout['state_action_values'][ix][:num_available_actions_ix]
                # Only adds an instance to the data set if the estimated action values are not all the same.
                if not np.allclose(action_values, action_values[0]):
                    choice_index = np.random.choice(np.flatnonzero(action_values == np.max(action_values)))
                    self.data_set.push(features=action_features,
                                       choice_index=choice_index,
                                       delete_oldest=False)
        self.v_print(f"self.current_number_of_choice_sets = {self.data_set.current_number_of_choice_sets}; "
                     f"len(self.data) = {len(self.data_set.data)}")

    def fit(self):
        if len(self.data_set.data) > 0:
            data_set = self.data_set.sample()
            if self.regularization and self.regularization_strength_method == "cv":
                policy_weights, _ = self.model.cv_fit(data=data_set, standardize=False)
            else:
                # Fixed lambda / regularization strength. If no regularization, lambda should be 0 (see init)
                policy_weights = self.model.fit(data=data_set,
                                                lam=self.regularization_strength,
                                                standardize=False)
            self.policy_weights = policy_weights
        return self.policy_weights

    @staticmethod
    def create_stew_matrix(num_features):
        D = np.full((num_features, num_features), fill_value=-1.0, dtype=np.float_)
        for i in range(num_features):
            D[i, i] = num_features - 1.0
        return D

    @staticmethod
    def create_ridge_matrix(num_features):
        D = np.eye(num_features)
        return D


class MultinomialLogisticRegressionWithDirections(MultinomialLogisticRegression):
    def __init__(
        self,
        feature_directions,
        num_features,
        max_choice_set_size,
        max_number_of_choice_sets=np.inf,
        regularization=False,
        regularization_method="stew",
        regularization_strength=0.0,
        regularization_strength_method="fixed",
        policy_weights=None
    ):
        super().__init__(
            num_features=num_features,
            max_choice_set_size=max_choice_set_size,
            max_number_of_choice_sets=max_number_of_choice_sets,
            regularization=regularization,
            regularization_method=regularization_method,
            regularization_strength=regularization_strength,
            regularization_strength_method=regularization_strength_method,
            policy_weights=policy_weights
        )
        self.feature_directions = feature_directions

    def fit(self, **rollout):
        if len(self.data_set.data) > 0:
            choice_data_set = self.data_set.sample()

            # Direct features according to the feature directions.
            choice_data_set[:, 2:] = choice_data_set[:, 2:] * self.feature_directions

            # Account for the fact that some directions have not been decided & delete
            # the corresponding features from the training data.
            non_zero_weights = np.where(self.feature_directions)[0]
            num_non_zero_weights = len(non_zero_weights)
            self.update_num_features_in_model(num_non_zero_weights)
            zero_weights = np.where(self.feature_directions == 0)[0]
            relevant_choice_data = np.delete(choice_data_set, obj=zero_weights + 2, axis=1)

            # Do the actual fitting
            if self.regularization and self.regularization_strength_method == "cv":
                policy_weights, _ = self.model.cv_fit(data=relevant_choice_data, standardize=False)
            else:
                # Fixed lambda / regularization strength. If no regularization, lambda should be 0 (see init)
                policy_weights = self.model.fit(data=relevant_choice_data,
                                                lam=self.regularization_strength,
                                                standardize=False)

            # Expand learned policy_weights with zeros for irrelevant features
            # and "un-direct" policy_weights
            self.policy_weights = np.zeros(self.num_features)
            self.policy_weights[non_zero_weights] = policy_weights
            self.policy_weights *= self.feature_directions
        return policy_weights

    def update_num_features_in_model(self, new_num_features):
        if self.regularization:
            if self.regularization_method == "stew":
                D = self.create_stew_matrix(new_num_features)
            elif self.regularization_method == "ridge":
                D = self.create_ridge_matrix(new_num_features)
        else:
            assert self.regularization_strength == 0.0, \
                "If regularization is turned off, you have to use regularization_strength == 0.0," \
                "which is also the default value!"
            D = np.zeros((new_num_features, new_num_features))

        self.model = StewMultinomialLogit(num_features=new_num_features, D=D)


class LearningFeatureDirections(PolicyApproximator):
    """
    Learning feature directions (LFD) algorithm to learn an equal-weighting policy made of
    feature direction estimates

        d_i \in {-1, 1}

    for i = 1, ..., num_features.
    """

    def __init__(
            self,
            num_features,
            max_choice_set_size,
            max_number_of_choice_sets=np.inf,
            verbose=False
    ):
        super().__init__(
            num_features=num_features,
            max_choice_set_size=max_choice_set_size,
            max_number_of_choice_sets=max_number_of_choice_sets,
            verbose=verbose
        )

        # policy_weights "=" feature directions
        self.policy_weights = np.zeros(self.num_features, dtype=np.float)
        self.positive_direction_counts = np.zeros(self.num_features)
        self.meaningful_comparisons = np.zeros(self.num_features)
        self.learned_directions = np.zeros(self.num_features)

    def append_data(self, **rollout):
        """
        This differs from MultinomialLogisticRegression.append_data() in that the rollout is not only
        added as a classification instance but is also used to "count" positive and negative
        associations of each feature with the rollout-decision outcome.

        :param rollout: a dictionary containing
            - state_action_features, a 3D NumPy array of floats of shape (self.rollout_set_size, self.num_actions, self.num_features)
            - state_action_values, a 2D NumPy array of floats of shape (self.rollout_set_size, self.num_actions)
            - num_available_actions, a 1D NumPy array of floats of size self.rollout_set_size
            - did_rollout, a 1D NumPy array of floats of size self.rollout_set_size
        :return:
        """
        for ix in range(len(rollout['state_action_values'])):
            if rollout['did_rollout'][ix]:
                num_available_actions_ix = rollout['num_available_actions'][ix]
                action_features = rollout['state_action_features'][ix][:num_available_actions_ix, :]
                action_values = rollout['state_action_values'][ix][:num_available_actions_ix]
                # Only adds an instance to the data set if the estimated action values are not all the same.
                if not np.allclose(action_values, action_values[0]):
                    choice_index = np.random.choice(np.flatnonzero(action_values == np.max(action_values)))
                    self.data_set.push(features=action_features,
                                       choice_index=choice_index,
                                       delete_oldest=False)
                    # Keep track of positive / negative associations.
                    chosen_action_features = action_features[choice_index]
                    remaining_action_features = np.delete(arr=action_features, obj=choice_index, axis=0)
                    feature_differences = np.sign(chosen_action_features - remaining_action_features)
                    direction_counts = np.sign(np.sum(feature_differences, axis=0))
                    self.positive_direction_counts += np.maximum(direction_counts, 0)
                    self.meaningful_comparisons += np.abs(direction_counts)

        self.v_print(f"self.current_number_of_choice_sets = {self.data_set.current_number_of_choice_sets}; "
                     f"len(self.data) = {len(self.data_set.data)}")

    def fit(self):
        """
        Checks for each feature whether its direction can be "decided with confidence", that is, whether its
        proportion of positive / negative associations with the response variable is significantly different
        from 50 / 50.

        :return: a NumPy array containing the new policy_weights
        """
        if len(self.data_set.data) > 0:
            unidentified_directions = np.where(self.learned_directions == 0.)[0]
            for feature_ix in range(len(unidentified_directions)):
                feature = unidentified_directions[feature_ix]
                p_value = binom_test(x=self.positive_direction_counts[feature],
                                     n=self.meaningful_comparisons[feature],
                                     p=0.5,
                                     alternative="two-sided")
                self.v_print(f"Feature {feature} has "
                             f"{self.positive_direction_counts[feature]} / {self.meaningful_comparisons[feature]} "
                             f"positive effective comparisons. P-value: {np.round(p_value, 4)}")

                if p_value < 0.1:
                    self.learned_directions[feature] = np.sign(
                        self.positive_direction_counts[feature] / self.meaningful_comparisons[feature] - 0.5)
                    self.v_print(f"Feature {feature} has been decided to be: {self.learned_directions[feature]}")

            self.policy_weights = self.learned_directions
        self.v_print(f"New policy_weights: {self.policy.policy_weights}")
        return self.policy_weights


