import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gym
from collections import deque
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from torch import nn
from torch.nn import functional


class Model(nn.Module):
    def __init__(self, input_features, output_values):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=input_features, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=output_values)

    def forward(self, x):
        x = functional.selu(self.fc1(x))
        x = functional.selu(self.fc2(x))
        x = self.fc3(x)
        return x



class DQN:
    def __init__(self):
        self.x = np.round(np.linspace(-0.3, 0.3, num=30), decimals=3)
        self.x_dot = np.round(np.linspace(-1.5, 1.5, num=15), decimals=3)
        self.th = np.round(np.linspace(-0.21, 0.21, num=21), decimals=3)
        self.th_dot = np.round(np.linspace(-2.5, 2.5, num=25), decimals=3)
        
        self.use_cuda = True
        self.learning_rate = 1e-4
        self.eps = 1
        self.gamma = 0.9
        self.memory_size = 1000

        self.env = gym.make('CartPole-v1')
        self.n_features = 4
        self.n_actions = 2

        self.memory=deque(maxlen=self.memory_size)

        self.device=torch.device('cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu')
        self.criterion=nn.MSELoss()
        self.policy_net=Model(self.n_features, self.n_actions).to(self.device)
        self.target_net=Model(self.n_features, self.n_actions).to(self.device)
        self.target_net.eval()

        self.optimizer=torch.optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)

    def get_state(self, desc):
        def nearest(arr, x):
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
        x = nearest(self.x, desc[0])
        x_dot = nearest(self.x_dot, desc[1])
        th = nearest(self.th, desc[2])
        th_dot = nearest(self.th_dot, desc[3])
        return (x, x_dot, th, th_dot)
    
    def get_reward(self, state, r):
        return r - state[0]**2 - state[2]**2
    
    def get_action(self, state):
        if np.random.randn()<self.eps:
            return np.random.choice([0,1])
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            return self.policy_net(state).argmax().item()
    
    def get_states_tensor(self, sample, states_idx):
        sample_len = len(sample)
        states_tensor = torch.empty((sample_len, self.n_features), dtype=torch.float32, requires_grad=False)

        features_range = range(self.n_features)
        for i in range(sample_len):
            for j in features_range:
                states_tensor[i, j] = sample[i][states_idx][j].item()

        return states_tensor
    
    def fit(self, X, y):
        X=X.to(self.device)
        y=y.to(self.device)
        train_ds=TensorDataset(X,y)
        train_dl=DataLoader(train_ds, batch_size=4)
        
        tot_loss=0

        for (input, label) in train_dl:
            out = self.policy_net(input)
            loss = self.criterion(out, label)
            tot_loss+=loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy_net.eval()
        # return tot_loss/len(X)

    def optimize_model(self, train_batch_size):
        batch_size=min(train_batch_size, len(self.memory))
        train_sample = np.random.choice(self.memory, batch_size)
        state = self.get_states_tensor(train_sample, 0)
        next_state = self.get_states_tensor(train_sample, 3)

        q_est=self.policy_net(state.to(self.device)).detach()
        next_state_q_est = self.target_net(next_state.to(self.device)).detach()

        for i in range(len(train_sample)):
            q_est[i][train_sample[i][1]]=self.get_reward(next_state[i], train_sample[i][2])+self.gamma*next_state_q_est[i].max()

        self.fit(state, q_est)
    
    def train_one_episode(self):
        cur_state, _ = self.env.reset()
        cur_state = self.get_state(cur_state)
        term, trunc=False, False
        score=0
        reward=0
        while not term and not trunc:
            action=self.get_action(cur_state)
            next_state, r, term, trunc, _ = self.env.step(action)
            next_state = self.get_state(next_state)
            cur_state=next_state
            score+=r
            reward+=self.get_reward(next_state, r)

            self.optimize_model(100)
        
        return score, reward
    
    def test(self):
        state, _=self.env.reset()
        state=self.get_state(state)
        term, trunc=False, False
        score=0
        reward=0
        while not term and not trunc:
            action=self.get_action(state)
            state, r, term, trunc, _ = self.env.step(action)
            state=self.get_state(state)
            score+=r
            reward+=self.get_reward(state, r)
        
        return score, reward    


def main():
    best_test_reward=0
    episode_cnt = 1000
    target_update_c = 10
    test_delay=10

    obj = DQN()
    train_score = []
    train_reward = []
    test_scoreh = []
    test_rewardh = []
    for i in tqdm(range(episode_cnt)):
        score, reward = obj.train_one_episode()
        train_score.append(score)
        train_reward.append(reward)
        # print(f'Episode {i + 1}: score: {score} - reward: {reward}')
        if i % target_update_c == 0:
            obj.target_net.load_state_dict(obj.policy_net.state_dict())
            obj.target_net.eval()

        if (i + 1) % test_delay == 0:
            test_score, test_reward = obj.test()
            test_scoreh.append(test_score)
            test_rewardh.append(test_reward)
            # print(f'Test Episode {i + 1}: test score: {test_score} - test reward: {test_reward}')
            if test_reward > best_test_reward:
                # print('New best test reward. Saving model')
                best_test_reward = test_reward
                torch.save(obj.policy_net.state_dict(), 'policy_net.pth')

    if episode_cnt % test_delay != 0:
        test_score, test_reward = obj.test()
        # print(f'Test Episode {episode_cnt}: test score: {test_score} - test reward: {test_reward}')
        if test_reward > best_test_reward:
            # print('New best test reward. Saving model')
            best_test_reward = test_reward
            torch.save(obj.policy_net.state_dict(), 'policy_net.pth')

    # print(f'best test reward: {best_test_reward}')
    plt.plot(range(len(train_score)), train_score)
    plt.show()
    plt.plot(range(len(train_reward)), train_reward)
    plt.show()

main()

