import time

import gym
import numpy as np

env = gym.make('Taxi-v3')

state_n = 500
action_n = 6


class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        p = self.model[state] / np.sum(self.model[state])
        action = np.random.choice(np.arange(self.action_n), p=p)
        return int(action)

    def fit(self, elite_trajectories, mode=None, settings=None):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(
                trajectory['states'], trajectory['actions']
            ):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                if mode == 'laplace':
                    laplace_lambda = settings['laplace_lambda']
                    new_model[state] += laplace_lambda
                    new_model[state] /= (
                        np.sum(new_model[state]) + laplace_lambda * action_n
                    )

                else:
                    new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        if mode == 'smoothie':
            smoothie_lambda = settings['smoothie_lambda']
            self.model = (
                smoothie_lambda * new_model + (1 - smoothie_lambda) * self.model
            )
        else:
            self.model = new_model
        return None


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(action)
        next_state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = obs

        if visualize:
            time.sleep(0.5)
            env.render()

        if done:
            break

    return trajectory


def run(best_iter_n, best_q_param, best_trajectory_n, mode=None, settings=None):
    global_rewards = []
    agent = CrossEntropyAgent(state_n, action_n)

    for iteraion in range(best_iter_n):
        # Policy Evaluation
        trajectories = [
            get_trajectory(env, agent) for _ in range(best_trajectory_n)
        ]
        total_rewards = [
            np.sum(trajectory['rewards']) for trajectory in trajectories
        ]

        # Policy Improvement
        elite_trajectories = []
        quantile = np.quantile(total_rewards, best_q_param)
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        global_rewards.append(total_rewards)

        agent.fit(elite_trajectories, mode, settings)

    trajectory = get_trajectory(env, agent, max_len=1000, visualize=False)

    print(
        f'Parameters: iteration_n {best_iter_n}, trajectory_n {best_trajectory_n}, q_param {best_q_param}'
    )
    print(f'Total reward: {sum(trajectory["rewards"])}')

    return global_rewards


if __name__ == '__main__':
    best_iter_n = 30
    best_q_param = 0.6
    best_trajectory_n = 400

    run(
        best_iter_n,
        best_q_param,
        best_trajectory_n,
        'laplace',
        {'laplace_lambda': 0.2},
    )
    run(
        best_iter_n,
        best_q_param,
        best_trajectory_n,
        'smoothie',
        {'smoothie_lambda': 0.2},
    )
