import torch
import torch.nn as nn  
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(128, output_dim)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x


model_path = "traffic_optimizer_model.pth"
agent = QNetwork(input_dim=3, output_dim=3)  
agent.load_state_dict(torch.load(model_path))
agent.eval()  

def optimize_traffic(state):
    
    state_tensor = torch.tensor(state, dtype=torch.float32)

    
    q_values = agent(state_tensor)

    
    epsilon = 0.9  
    if np.random.rand() < epsilon:  
        action = np.random.randint(0, 3)
    else:  
        action = torch.argmax(q_values).item()

    return action


def adjust_network_parameters(action):
    
    if action == 0:
        print("Action 0: Reduce latency")
    elif action == 1:
        print("Action 1: Increase throughput")
    elif action == 2:
        print("Action 2: Balance network load")


state = [5, 2, 3]


action = optimize_traffic(state)


adjust_network_parameters(action)

import time

while True:
    state = [np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 5)]  
    action = optimize_traffic(state)

    adjust_network_parameters(action)

    time.sleep(1)

