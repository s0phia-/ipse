from agents.rollout_mechanisms import BatchRollout
from agents.policy_approximators import MultinomialLogisticRegressionWithDirections, \
    LearningFeatureDirections
from agents.rbpi import Rbpi

import numpy as np


class Ipse(Rbpi):
    """
    Iterative policy-space expansion

    Like Rbpi but switches the policy approximator / "phase" based on some conditions,
    which are checked after every learning step (see self.learn()).

    """
    def __init__(
        self,
        env,
        first_phase_policy_approximator,
        second_phase_policy_approximator,
        rollout_handler,
        forced_phase_switch_after=15,
    ):
        super().__init__(
            env=env,
            policy_approximator=second_phase_policy_approximator,
            rollout_handler=rollout_handler
        )
        self.policy = first_phase_policy_approximator
        self.second_phase_policy = second_phase_policy_approximator
        self.current_phase = 1
        self.forced_phase_switch_after = forced_phase_switch_after
        self.iteration_in_current_phase = 0

    def learn(self, *args, **kwargs):
        # Use parent-learn() to do rollouts and self.policy.fit()
        super().learn(args, kwargs)
        self.iteration_in_current_phase += 1

        # Check whether "phase" should be changed (and if yes, make the change)
        # Currently the phase switches from phase 1 to phase 2
        # a) after a hard-coded amount (self.forced_phase_switch_after) of iterations and
        #    if at least one feature direction has been learned, or
        # b) after all feature directions have been learned.
        if (
                (
                    self.current_phase == 1 and
                    self.iteration_in_current_phase >= self.forced_phase_switch_after and
                    self.policy.policy_weights.any()
                )
            or
                (
                    self.current_phase == 1 and
                    np.all(self.policy.policy_weights != 0.0)
                )
        ):
            print("Switching phase!")
            self.current_phase += 1
            self.iteration_in_current_phase = 0

            # Store feature directions and data already collected...
            feature_directions = self.policy.policy_weights
            data_set = self.policy.data_set

            # ...and transfer them to the new policy approximator.
            self.policy = self.second_phase_policy
            self.policy.policy_weights = feature_directions
            self.policy.feature_directions = feature_directions
            self.policy.data_set = data_set


class ContinualIpse(Rbpi):
    """
    Version of Iterative policy-space expansion, where the
    algorithm does not wait for all feature directions
    to be decided before starting to learn linear models.

    There is no switch between policy approximation architectures
    anymore (this is the case in Ipse()). Instead, there is
    only one self.policy (e.g., MultinomialLogisticRegression)
    which starts off having zero weights.

    The self.learning_feature_direction_policy updates/informs
    self.policy whenever a new feature direction is learned.
    """
    def __init__(
            self,
            env,
            direction_learner,
            policy,
            rollout_handler,
    ):
        super().__init__(
            env=env,
            policy_approximator=policy,
            rollout_handler=rollout_handler
        )
        self.policy = policy
        self.direction_learner = direction_learner

    def learn(self, *args, **kwargs):
        """
        TODO

        Perform a rollout and add relevant information to the data set.
        As long as not all feature directions are decided:
            - update positive/negative feature associations
            - check whether feature directions can be decided
            - pass feature direction information to self.policy
            As soon as there is at least one feature direction:1??1?? m,.
                - fit a linear policy to the data for all features that
                  have been "directed".
        When all feature directions haven been decided:
            - fit a linear policy on all directed features
        """
        # TODO

        # # Rollouts
        # rollout = self.rollout_handler.perform_rollouts(self.policy)
        #
        # # Append rollout data to policy.data_set
        # self.policy.append_data(**rollout)


def create_ipse(env, parameter_dict, rollout_state_population):
    p = parameter_dict

    rollout_handler = BatchRollout(
        env=env,
        stochastic_rollout_policy=p.stochastic_rollout_policy,
        rollout_length=p.rollout_length,
        rollouts_per_action=p.rollouts_per_action,
        rollout_set_size=p.rollout_set_size,
        rollout_state_population=rollout_state_population
    )

    lfd = LearningFeatureDirections(
        num_features=env.num_features,
        max_choice_set_size=env.num_actions,
        max_number_of_choice_sets=p.max_number_of_choice_sets
    )

    policy_approximator = MultinomialLogisticRegressionWithDirections(
        feature_directions=np.zeros(env.num_features),
        num_features=env.num_features,
        max_choice_set_size=env.num_actions,
        max_number_of_choice_sets=p.max_number_of_choice_sets,
        regularization=p.regularization,
        regularization_method=p.regularization_method,
        regularization_strength=p.regularization_strength,
        regularization_strength_method=p.regularization_strength_method
    )

    agent = Ipse(
        env=env,
        first_phase_policy_approximator=lfd,
        second_phase_policy_approximator=policy_approximator,
        rollout_handler=rollout_handler,
        forced_phase_switch_after=p.forced_phase_switch_after
    )
    return agent

