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


def run(best_iter_n, best_trajectory_n, M):
    agent = CrossEntropyAgent(state_n, action_n)

    global_rewards = []
    for iteration in range(best_iter_n):
        total_rewards = []

        stochastic_policy = agent.model.copy()
        trajectories = []

        for _ in range(M):
            determ_policy = np.zeros((agent.state_n, agent.action_n))

            for state in range(agent.state_n):
                action = np.random.choice(
                    np.arange(agent.action_n), p=stochastic_policy[state]
                )
                determ_policy[state][action] += 1

            agent.model = determ_policy
            trs = [get_trajectory(env, agent) for _ in range(best_trajectory_n)]
            trajectories = trajectories + trs
            local_reward = np.mean(
                [np.sum(trajectory['rewards']) for trajectory in trs]
            )
            total_rewards += [local_reward for _ in trs]

        global_rewards.append(np.mean(total_rewards))
        agent.model = stochastic_policy

    return global_rewards


if __name__ == '__main__':
    best_iter_n = 30
    best_q_param = 0.6
    best_trajectory_n = 400
    M = 50

    run(best_iter_n, best_trajectory_n, M)
