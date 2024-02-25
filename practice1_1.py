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

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(
                trajectory['states'], trajectory['actions']
            ):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

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


def grid_search_agent_parametrs(
    iterations_grid, trajectories_grid, q_params_grid
):
    for iteration_n in iterations_grid:
        for trajectory_n in trajectories_grid:
            for q_param in q_params_grid:
                agent = CrossEntropyAgent(state_n, action_n)

                for iteraion in range(iteration_n):
                    # Policy Evaluation
                    trajectories = [
                        get_trajectory(env, agent) for _ in range(trajectory_n)
                    ]
                    total_rewards = [
                        np.sum(trajectory['rewards'])
                        for trajectory in trajectories
                    ]

                    # Policy Improvement
                    elite_trajectories = []
                    quantile = np.quantile(total_rewards, q_param)
                    for trajectory in trajectories:
                        total_reward = np.sum(trajectory['rewards'])
                        if total_reward > quantile:
                            elite_trajectories.append(trajectory)

                    agent.fit(elite_trajectories)

                trajectory = get_trajectory(
                    env, agent, max_len=1000, visualize=False
                )

                print(
                    f'Parameters: iteration_n {iteration_n}, trajectory_n {trajectory_n}, q_param {q_param}'
                )
                print(f'Total reward: {sum(trajectory["rewards"])}')


if __name__ == '__main__':
    # Задание 1
    # Пользуясь алгоритмом Кросс-Энтропии обучить агента решать задачу Taxi-v3 из Gym.
    # Исследовать гиперпараметры алгоритма и выбрать лучшие.

    iterations_grid = [10, 15, 20, 25]
    trajectories_grid = [100, 200, 300, 400]
    q_params_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

    grid_search_agent_parametrs(
        iterations_grid, trajectories_grid, q_params_grid
    )
