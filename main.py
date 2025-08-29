'''
# Group Number: 24
# Roll Numbers (Name of the member): 20IE10028 (Sai Lokesh Gorantla)
# Project Code: CP
# Project Title: Training a Cart Pole agent using Reinforcement Learning
'''


import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm as tq
from PIL import Image
import PIL.ImageDraw as ImageDraw
import imageio
import os
import json
import pickle


class Policy_Iterator:
    def __init__(self, states_path=None, training=True):
        self.x = np.round(np.linspace(-0.3, 0.3, num=30), decimals=3)
        self.x_dot = np.round(np.linspace(-1.5, 1.5, num=15), decimals=3)
        self.th = np.round(np.linspace(-0.21, 0.21, num=21), decimals=3)
        self.th_dot = np.round(np.linspace(-2.5, 2.5, num=25), decimals=3)
        self.env = gym.make(
            'CartPole-v1', render_mode='human' if training == False else None)
        self.training = training
        self.states_path = states_path
        if not os.path.exists(self.states_path):
            print("specified states file path does not exist")
        self.states = []
        self.actions = np.array([0, 1])
        self.policy = 1
        self.gamma = 0.8
        self.name = "Policy_iteration"
        if not self.training:
            self.load_states()
        else:
            self.generate_states()

    def generate_states(self):
        for x in self.x:
            for x_dot in self.x_dot:
                for th in self.th:
                    for th_dot in self.th_dot:
                        self.states.append((x, x_dot, th, th_dot))
        self.states = {state: {'actions': {0: 0.5, 1: 0.5},
                               'val': 0,
                               'next_state': {0: 0, 1: 0},
                               'reward': 0} for state in self.states}

    def nearest(self, arr, x):
        if x <= arr[0]:
            return arr[0]
        elif x >= arr[-1]:
            return arr[-1]
        n = len(arr)
        lo, hi, mid = 0, n-1, 0
        while (lo < hi):
            mid = lo+int((hi-lo)/2)
            if arr[mid] == x:
                return arr[mid]
            if x < arr[mid]:
                if (mid > 0 and x > arr[mid - 1]):
                    return arr[mid] if arr[mid]-x >= arr[mid-1]-x else arr[mid-1]
                hi = mid
            else:
                if (mid < n - 1 and x < arr[mid + 1]):
                    return arr[mid] if arr[mid]-x >= arr[mid+1]-x else arr[mid+1]
                lo = mid + 1
        return arr[mid]

    def get_state(self, desc):
        x = self.nearest(self.x, desc[0])
        x_dot = self.nearest(self.x_dot, desc[1])
        th = self.nearest(self.th, desc[2])
        th_dot = self.nearest(self.th_dot, desc[3])
        return (x, x_dot, th, th_dot)

    def store_states(self):
        with open(self.states_path, 'wb') as f:
            pickle.dump(self.states, f)

    def load_states(self):
        with open(self.states_path, 'rb') as f:
            self.states = pickle.load(f)

    def get_next_states(self, file_path=None):
        if file_path is not None and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.states = pickle.load(f)
                return
        for state in tq(self.states.keys()):
            for action in self.actions:
                self.env.reset()
                self.env.state = state
                obs, reward, _, _, _ = self.env.step(
                    action)
                next_state = self.get_state(obs)
                self.states[state]['next_state'][action] = next_state
                self.states[state]['reward'] = -np.sum(np.square(next_state))
        self.store_states()

    def validate_states(self):
        for state in self.states:
            if self.states[state]['next_state'][0] == state or self.states[state]['next_state'][1] == state:
                print("The state", state, "is invalid")
                print(self.states[state])
                return
        print("All states are valid")

    def policy_iteration_1_step(self):
        # Evaluate current policy
        tmp = {}
        for state in self.states:
            reward = self.states[state]['reward']
            tmp[state] = self.states[state]
            t = 0
            for action in self.actions:
                # Probability of taking action a in state s
                pi_a_s = self.states[state]['actions'][action]
                next_state = self.states[state]['next_state'][action]
                v_s_dash = self.states[next_state]['val']
                t += (pi_a_s * (reward + self.gamma * v_s_dash))
            tmp[state]['val'] = t
        self.states = tmp

        # Improve policy
        changed = False
        for state in self.states:
            n1 = self.states[state]['next_state'][0]
            n2 = self.states[state]['next_state'][1]
            n1 = self.states[n1]['val']
            n2 = self.states[n2]['val']
            prev = self.states[state]['actions']
            if n1 >= n2:
                self.states[state]['actions'] = {0: 1, 1: 0}
                changed = changed or prev != {0: 1, 1: 0}
            else:
                self.states[state]['actions'] = {0: 0, 1: 1}
                changed = changed or prev != {0: 0, 1: 1}
        return changed

    def policy_iteration(self, epochs=200):
        changed = True
        for epoch in tq(range(epochs)):
            changed = self.policy_iteration_1_step()
            epochs -= 1
            if not changed:
                break
        print("Epochs left:", epochs)
        self.store_states()

    def get_action(self, state):
        state = self.get_state(state)
        n1 = self.states[state]['next_state'][0]
        n2 = self.states[state]['next_state'][1]
        n1 = self.states[n1]['val']
        n2 = self.states[n2]['val']
        return 0 if n1 >= n2 else 1


class MCLearning:
    def __init__(self, eps=0.1, use_pretrained=True):
        self.x = np.round(np.linspace(-0.3, 0.3, num=30), decimals=3)
        self.x_dot = np.round(np.linspace(-1.5, 1.5, num=15), decimals=3)
        self.th = np.round(np.linspace(-0.21, 0.21, num=21), decimals=3)
        self.th_dot = np.round(np.linspace(-2.5, 2.5, num=25), decimals=3)
        self.env = gym.make('CartPole-v1')
        self.actions = np.array([0, 1])
        self.run_hist = []
        self.eps = eps
        self.name = "Monte_carlo"
        if use_pretrained:
            self.policy = np.load('./MCLearning/custom_reward_1/policy.npy')
            self.st_act_val = np.load(
                './MCLearning/custom_reward_1/st_act_val.npy')
            with open('./MCLearning/custom_reward_1/st_dict_idx.pkl', 'rb') as f:
                self.st_dict_idx = pickle.load(f)
            with open('./MCLearning/custom_reward_1/st_idx.pkl', 'rb') as f:
                self.st_idx_dict = pickle.load(f)
            with open('./MCLearning/custom_reward_1/run_hist.pkl', 'rb') as f:
                self.run_hist = pickle.load(f)
            self.state_count = self.policy.shape[0]
            print("Resuming training from", len(self.run_hist), "episode!")

    def generate_states(self):
        self.states = []
        for x in self.x:
            for x_dot in self.x_dot:
                for th in self.th:
                    for th_dot in self.th_dot:
                        self.states.append((x, x_dot, th, th_dot))
        '''
        The structure of self.state_dict_idx: key is the state tuple, value is its index
        The structure of self.state_idx_dict
        '''
        self.st_dict_idx = {state: i for i, state in enumerate(self.states)}
        self.st_idx_dict = {i: state for i, state in enumerate(self.states)}
        '''
        The variable self.policy states the best action for each state, taking the state's index
        '''
        self.state_count = len(self.states)
        self.policy = np.random.randint(0, 2, (self.state_count))

        '''
        The structure of self.st_act_val is as follows:
        1. The length of this array depicts the config of each state.
        2. The two vectors in each state correspond to the two actions - 0, 1
        3. The first value for each action vector is the action value for that state
        4. The second value for each action vector is the freq with which that action is taken
        '''
        self.st_act_val = np.zeros((self.state_count, 2, 2))
        del self.states

    def get_state(self, desc):
        def nearest(v, x):
            if x <= v[0]:
                return v[0]
            elif x >= v[-1]:
                return v[-1]
            n = len(v)
            lo, hi, mid = 0, n-1, 0
            while (lo < hi):
                mid = lo+int((hi-lo)/2)
                if v[mid] == x:
                    return v[mid]
                if x < v[mid]:
                    if (mid > 0 and x > v[mid - 1]):
                        return v[mid] if v[mid]-x >= v[mid-1]-x else v[mid-1]
                    hi = mid
                else:
                    if (mid < n - 1 and x < v[mid + 1]):
                        return v[mid] if v[mid]-x >= v[mid+1]-x else v[mid+1]
                    lo = mid + 1
            return v[mid]
        x = nearest(self.x, desc[0])
        x_dot = nearest(self.x_dot, desc[1])
        th = nearest(self.th, desc[2])
        th_dot = nearest(self.th_dot, desc[3])
        return self.st_dict_idx[(x, x_dot, th, th_dot)]

    # Using e-greedy to evaluate policy
    def get_action(self, state):
        rn = np.random.uniform(0, 1)
        act_best = self.policy[self.get_state(state)]
        if rn > self.eps:
            return act_best
        else:
            return np.random.choice([0, 1])

    def save_agent(self):
        np.save('./MCLearning/custom_reward_1/policy.npy', self.policy)
        np.save('./MCLearning/custom_reward_1/st_act_val.npy', self.st_act_val)
        with open('./MCLearning/custom_reward_1/st_dict_idx.pkl', 'wb') as f:
            pickle.dump(self.st_dict_idx, f)
        with open('./MCLearning/custom_reward_1/st_idx.pkl', 'wb') as f:
            pickle.dump(self.st_idx_dict, f)
        with open('./MCLearning/custom_reward_1/run_hist.pkl', 'wb') as f:
            pickle.dump(self.run_hist, f)
        plt.plot(self.run_hist)
        plt.savefig('./MCLearning/custom_reward_1/run_hist.png')

    # Using every time visit
    def train(self, episodes=20):
        self.state_freq = np.zeros(self.state_count)
        for episode in tq(range(episodes)):
            history = []
            obs, _ = self.env.reset()
            while (1):
                action = self.get_action(obs)
                state = self.get_state(obs)
                obs, reward, term, trunc, _ = self.env.step(action)
                reward += (1-obs[0]**2-obs[2]**2)
                history.append((state, action, reward))
                if term or trunc:
                    break

            self.run_hist.append(len(history))

            cum_reward = 0
            for step in reversed(history):
                st, act, rew = step
                cum_reward += rew
                self.st_act_val[st][act][1] += 1
                self.st_act_val[st][act][0] += (
                    cum_reward - self.st_act_val[st][act][0])/self.st_act_val[st][act][1]

            ### Policy Improvement ###
            for st in range(self.state_count):
                tmp = np.array([self.st_act_val[st][0][0],
                               self.st_act_val[st][1][0]])
                self.policy[st] = np.argmax(tmp)
            if episode % 100 == 99:
                self.save_agent()
        self.save_agent()


class Sarsa:
    def __init__(self, gamma=0.9, lamda_=1, alpha=0.5, eps=0.1, use_pretrained=True):
        self.x = np.round(np.linspace(-0.3, 0.3, num=30), decimals=3)
        self.x_dot = np.round(np.linspace(-1.5, 1.5, num=15), decimals=3)
        self.th = np.round(np.linspace(-0.21, 0.21, num=21), decimals=3)
        self.th_dot = np.round(np.linspace(-2.5, 2.5, num=25), decimals=3)
        self.env = gym.make('CartPole-v1')
        self.actions = np.array([0, 1])
        self.alpha = alpha
        self.run_hist = []
        self.eps = eps
        self.gamma = gamma
        self.lambda_ = lamda_
        self.name = "Sarsa"

        if use_pretrained:
            self.policy = np.load('./TDLearning/policy.npy')
            self.st_act_val = np.load('./TDLearning/st_act_val.npy')
            with open('./TDLearning/st_dict_idx.pkl', 'rb') as f:
                self.st_dict_idx = pickle.load(f)
            with open('./TDLearning/st_idx.pkl', 'rb') as f:
                self.st_idx_dict = pickle.load(f)
            with open('./TDLearning/run_hist.pkl', 'rb') as f:
                self.run_hist = pickle.load(f)
            self.state_count = self.policy.shape[0]
            print("Resuming training from", len(self.run_hist), "episode!")

    def generate_states(self):
        self.states = []
        for x in self.x:
            for x_dot in self.x_dot:
                for th in self.th:
                    for th_dot in self.th_dot:
                        self.states.append((x, x_dot, th, th_dot))
        '''
        The structure of self.state_dict_idx: key is the state tuple, value is its index
        The structure of self.state_idx_dict
        '''
        self.st_dict_idx = {state: i for i, state in enumerate(self.states)}
        self.st_idx_dict = {i: state for i, state in enumerate(self.states)}
        '''
        The variable self.policy states the best action for each state, taking the state's index
        '''
        self.state_count = len(self.states)
        self.policy = np.random.randint(0, 2, (self.state_count))

        '''
        The structure of self.st_act_val is as follows:
        1. The length of this array depicts the config of each state.
        2. The two vectors in each state correspond to the two actions - 0, 1
        3. The first value for each action vector is the action value for that state
        4. The second value for each action vector is the freq with which that action is taken
        '''
        self.st_act_val = np.zeros((self.state_count, 2, 2))
        del self.states

    def get_state(self, desc):
        def nearest(v, x):
            if x <= v[0]:
                return v[0]
            elif x >= v[-1]:
                return v[-1]
            n = len(v)
            lo, hi, mid = 0, n-1, 0
            while (lo < hi):
                mid = lo+int((hi-lo)/2)
                if v[mid] == x:
                    return v[mid]
                if x < v[mid]:
                    if (mid > 0 and x > v[mid - 1]):
                        return v[mid] if v[mid]-x >= v[mid-1]-x else v[mid-1]
                    hi = mid
                else:
                    if (mid < n - 1 and x < v[mid + 1]):
                        return v[mid] if v[mid]-x >= v[mid+1]-x else v[mid+1]
                    lo = mid + 1
            return v[mid]
        x = nearest(self.x, desc[0])
        x_dot = nearest(self.x_dot, desc[1])
        th = nearest(self.th, desc[2])
        th_dot = nearest(self.th_dot, desc[3])
        return self.st_dict_idx[(x, x_dot, th, th_dot)]

    # Using e-greedy to evaluate policy
    def get_action(self, state):
        rn = np.random.uniform(0, 1)
        act_best = self.policy[self.get_state(state)]
        if rn > self.eps:
            return act_best
        else:
            return np.random.choice([0, 1])

    def save_agent(self):
        np.save('./TDLearning/policy.npy', self.policy)
        np.save('./TDLearning/st_act_val.npy', self.st_act_val)
        with open('./TDLearning/st_dict_idx.pkl', 'wb') as f:
            pickle.dump(self.st_dict_idx, f)
        with open('./TDLearning/st_idx.pkl', 'wb') as f:
            pickle.dump(self.st_idx_dict, f)
        with open('./TDLearning/run_hist.pkl', 'wb') as f:
            pickle.dump(self.run_hist, f)
        plt.plot(self.run_hist)
        plt.savefig('./TDLearning/run_hist.png')

    def train(self, episodes=20):
        self.state_freq = np.zeros(self.state_count)
        for episode in tq(range(episodes)):
            eligibility = np.zeros((self.state_count, 2))
            obs, _ = self.env.reset()
            a = self.get_action(obs)
            s = self.get_state(obs)
            active_time = 1
            while (1):
                obs, reward, term, trunc, _ = self.env.step(a)
                sd = self.get_state(obs)
                ad = self.get_action(obs)
                # reward += -np.sum(np.log(np.abs(obs)))
                reward += (1-obs[0]**2-obs[2]**2)

                delta = reward + self.gamma * \
                    self.st_act_val[sd][ad][0] - self.st_act_val[s][a][0]
                eligibility[s][a] += 1
                for state in range(self.state_count):
                    for action in [0, 1]:
                        self.st_act_val[state][action][0] += self.alpha * delta * \
                            eligibility[state][action]
                        eligibility[state][action] *= self.gamma * self.lambda_
                s = sd
                a = ad
                active_time += 1

                if term or trunc:
                    break
            self.run_hist.append(active_time)
            if episode % 100 == 99:
                self.save_agent()
            self.save_agent()


def main(agent):
    def _label_with_episode_number(frame, episode_num, action, step):
        im = Image.fromarray(frame)

        drawer = ImageDraw.Draw(im)

        if np.mean(im) < 128:
            text_color = (255,255,255)
        else:
            text_color = (0,0,0)
        drawer.text((im.size[0]/20,50), f'Episode: {episode_num+1}', fill=text_color)
        drawer.text((im.size[0]/20,70), f'Action: {action}', fill=text_color)
        drawer.text((im.size[0]/20,90), f'Time Step: {step}', fill=text_color)

        return im


    def save_random_agent_gif(env):
        frames = []
        avg, mx = 0, 0
        for episode in tq(range(10)):
            obs, _ = env.reset()
            step=0
            while(1):
                action = agent.get_action(obs)
                frame = env.render()
                frames.append(_label_with_episode_number(frame, episode_num=episode, action=action, step=step))
                step+=1
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            avg += step
            mx = max(mx, step)
        env.close()
        print("Average # time steps active for:", avg/10)
        print("Maximum # time steps active for:", mx)
        writer = imageio.get_writer(os.path.join('./videos/', agent.name+'.mp4'), fps=60)
        for im in frames:
            writer.append_data(np.array(im))
        writer.close()


    env = gym.make('CartPole-v1', render_mode='rgb_array')
    save_random_agent_gif(env)

''' 
THE FOLLOWING CODE BLOCK IS FOR VISUALIZATION PURPOSES ONLY
1. Uncomment the code block corresponding to which ever agent is desired to be visualized.
2. The main function call should remain uncommented to generate the video for each of the agents the agents.
3. After the code is finished running, the corresponding video will be saved in the `./videos` folder.
4. During training, the following code block (consisting of four lines) should be commented out.
'''
# agent = Policy_Iterator('./policy_iterator.pkl', training=False)
# main(agent)

# agent = MCLearning(eps=0.2, use_pretrained=True)
# main(agent)

# agent = Sarsa(gamma=0.9, lamda_=1, alpha=0.5, eps=0, use_pretrained=True)
# main(agent)


'''
THE FOLLOWING CODE BLOCK IS FOR VISUALIZATION PURPOSES ONLY
1. Uncomment the respective code block to train the agent under the desired policy.
2. If the epochs left after training the agent using Policy iteration is 0, it means we need to train the agent further.
3. Mention the name of the file to be generated for policy iteration after the training is done.
4. The optimal policy is saved for every 100 episodes in Monte Carlo and SARSA learning algorithms.
'''
# agent = Policy_Iterator('./test.pkl', training=True)
# agent.get_next_states()
# agent.policy_iteration(epochs=500)

agent = MCLearning(eps=0.3, use_pretrained=False)
agent.generate_states()
agent.train(episodes=10000)

# agent = Sarsa(gamma=0.9, lamda_=1, alpha=0.5, eps=0.1, use_pretrained=False)
# agent.generate_states()
# agent.train(episodes=500)