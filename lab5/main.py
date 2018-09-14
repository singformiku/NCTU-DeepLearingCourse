import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from statistics import mean

# hyper parameters
EPISODES = 1001  # number of episodes
EPS_START = 1  # e-greedy threshold start value
EPS_END = 0.01  # e-greedy threshold end value
EPS_DECAY = 0.995  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.0005 # NN optimizer learning rate
HIDDEN_LAYER = 32  # NN hidden layer size
BATCH_SIZE = 128  # Q-learning batch size
TARGET_UPDATE = 50

# use GPU
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


env = gym.make('CartPole-v0').unwrapped
#env = wrappers.Monitor(env, './tmp/cartpole-v0-1')

Target_Net = Network()
Target_Net.cuda()
Action_Net = Network()
Action_Net.cuda()
memory = ReplayMemory(5000)
optimizer = optim.Adam(Action_Net.parameters(), LR)
steps_done = 0
episode_durations = []

def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.ylim(0,400)
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def select_action(state):
    global steps_done
    global EPS_START
    sample = random.random()
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = EPS_START*EPS_DECAY
    EPS_START = eps_threshold
    if eps_threshold<EPS_END:
        eps_threshold=EPS_END
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return Action_Net(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

max_pt = 195
def run_episode(e, environment):
    global max_pt 
    state = environment.reset()
    steps = 0
    while True:
        environment.render()
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action.item())

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done:
            print("Episode {0} finished after {1} steps".format(e, steps))
            episode_durations.append(steps)
            plot_durations()
            if steps > max_pt:
                max_pt = steps
                torch.save(Action_Net.state_dict(), 'act_net.pth')
            break


test=[]
def testing(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        environment.render()
        #Action_Net.load_state_dict(torch.load('act_net.pth'))
        with torch.no_grad():
            action = Action_Net(Variable(FloatTensor([state]))).data.max(1)[1].view(1, 1)
        next_state, reward, done, _ = environment.step(action.item())

        # negative reward when attempt ends
        if done:
            reward = -1

        state = next_state
        steps += 1

        if done:
            print("Testing {0} finished after {1} steps".format(e, steps))
            test.append(steps)
            break


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions

    current_q_values = Action_Net(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = Target_Net(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    expected_q_values.resize_(expected_q_values.size(0), 1)
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
Action_Net.load_state_dict(torch.load('act_net2000.pth'))
Target_Net.load_state_dict(Action_Net.state_dict())
for e in range(EPISODES):   #training
    run_episode(e, env)
    if e % TARGET_UPDATE == 0:
        Target_Net.load_state_dict(Action_Net.state_dict())

for e in range(100):   #testing for 100 times
    testing(e, env)
print ("Avg testing reward: {0}".format(mean(test)))

env.render()
env.close()
plt.savefig("reward-1.png")
plt.show()