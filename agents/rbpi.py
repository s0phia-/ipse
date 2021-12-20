from agents.rollout_mechanisms import BatchRollout
from agents.policy_approximators import MultinomialLogisticRegression


class Rbpi:
    """
    Rollout-based policy iteration
    """
    def __init__(self,
                 env,
                 policy_approximator,
                 rollout_handler
                 ):
        self.policy = policy_approximator
        self.env = env
        self.rollout_handler = rollout_handler

        self.num_features = self.policy.num_features
        self.policy_weights = policy_approximator.policy_weights

    def learn(self, *args, **kwargs):
        # Rollouts
        rollout = self.rollout_handler.perform_rollouts(self.policy)

        # Append rollout data to policy.data_set
        self.policy.append_data(**rollout)

        # Policy approximation
        self.policy.fit()

    def choose_action(self, choice_set, stochastic=False):
        return self.policy.choose_action(choice_set, stochastic=stochastic)


def create_rbpi(env, parameter_dict, rollout_state_population):
    p = parameter_dict

    rollout_handler = BatchRollout(
        env=env,
        stochastic_rollout_policy=p.stochastic_rollout_policy,
        rollout_length=p.rollout_length,
        rollouts_per_action=p.rollouts_per_action,
        rollout_set_size=p.rollout_set_size,
        rollout_state_population=rollout_state_population
    )

    policy_approximator = MultinomialLogisticRegression(
        num_features=env.num_features,
        max_choice_set_size=env.num_actions,
        max_number_of_choice_sets=p.max_number_of_choice_sets,
        regularization=p.regularization,
        regularization_method=p.regularization_method,
        regularization_strength=p.regularization_strength,
        regularization_strength_method=p.regularization_strength_method
    )

    agent = Rbpi(
        env=env,
        policy_approximator=policy_approximator,
        rollout_handler=rollout_handler
    )
    return agent

