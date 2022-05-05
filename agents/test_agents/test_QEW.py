import unittest
from agents.QEW import *
from gym.spaces import Discrete


class TestQEW(unittest.TestCase):

    def test_fit_closed_form_1(self):
        """
        check both sides match
        """
        agent = QEW(num_features=4, actions=Discrete(2))
        agent.X = np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                            [5, 6, 7, 8, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 2, 3, 4],
                            [0, 0, 0, 0, 5, 6, 7, 8]])
        agent.y = np.array([3, 5, 3, 5])
        agent.fit_closed_form()
        np.testing.assert_array_almost_equal(agent.beta[:agent.num_features], agent.beta[agent.num_features:])

    def test_fit_closed_form_2(self):
        """
        Check known solution
        """
        agent = QEW(num_features=8, actions=Discrete(5))
        np.random.seed(102)
        quadrant_1 = np.random.rand(10, 20)
        quadrant_2 = np.zeros([10, 20])
        quadrant_3 = quadrant_2
        np.random.seed(103)
        quadrant_4 = np.random.rand(10, 20)
        half_1 = np.hstack((quadrant_1, quadrant_2))
        half_2 = np.hstack((quadrant_3, quadrant_4))
        agent.X = np.vstack((half_1, half_2))
        agent.y = np.random.rand(20)
        agent.lam = 0.45
        beta = [0.04335684, 0.04360118, 0.04379954, 0.04338769, 0.04416202,
                0.04357364, 0.04376297, 0.04293298, 0.04381937, 0.04357809,
                0.04404939, 0.04334311, 0.04337596, 0.04326849, 0.04355304,
                0.04344566, 0.0428718, 0.04340834, 0.04323131, 0.04334417,
                0.04311754, 0.04316623, 0.04302911, 0.04313675, 0.04269931,
                0.04312515, 0.04266119, 0.04302607, 0.04339114, 0.04316945,
                0.04279915, 0.04269564, 0.04270336, 0.04281799, 0.04295812,
                0.04307172, 0.04291723, 0.04307197, 0.04293672, 0.04270912]
        # np.matmul(np.linalg.inv(np.matmul(agent.X.transpose(), agent.X) +
        #                                   agent.lam*np.matmul(agent.D.transpose(), agent.D)),
        #           np.matmul(agent.X.transpose(), agent.y))
        agent.fit_closed_form()
        np.testing.assert_array_almost_equal(agent.beta, beta)

    def test_get_highest_q_action(self):
        """
        The highest weighted features are chosen
        """
        agent = QEW(num_features=6, actions=Discrete(3))
        agent.beta = np.array([8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2])
        action, _ = agent.get_highest_q_action(np.random.rand(6))
        self.assertEqual(action, 0)

    def test_append_data(self):
        """
        The updated...
        """
        agent = QEW(5, Discrete(3))
        agent.experience_window = 100
        state_features = [.1, .2, .3, .4, .5]
        action = 2
        reward = 0
        state_prime_features = [0, 0, 0, 0, 0]
        agent.X = np.zeros([100, 15])
        agent.y = np.zeros([100, 1])
        agent.store_data(state_features, action, reward, state_prime_features)
        expected_x = np.vstack([np.zeros([99, 15]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .1, .2, .3, .4, .5]])
        expected_y = np.zeros([100])
        np.testing.assert_array_equal(agent.X, expected_x)
        np.testing.assert_array_equal(agent.y, expected_y)


if __name__ == '__main__':
    unittest.main()
