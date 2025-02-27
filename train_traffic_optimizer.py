import torch
import torch.nn as nn
import torch.optim as optim
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

class TrafficEnvironment:
    def __init__(self):
        self.state = np.array([0, 0, 0])  
        self.done = False  
    
    def reset(self):
        self.state = np.array([0, 0, 0])  
        self.done = False
        return self.state
    
    def step(self, action):
        self.state = self.state + action  
        reward = -np.sum(action)  
        
        if np.sum(self.state) > 10:  
            self.done = True
        return self.state, reward, self.done

input_dim = 3  
output_dim = 3  
learning_rate = 0.001
gamma = 0.99  

env = TrafficEnvironment()
agent = QNetwork(input_dim, output_dim)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for episode in range(1000):  
    state = torch.tensor(env.reset(), dtype=torch.float32)  
    done = False
    
    while not done:
        q_values = agent(state)  
        epsilon = 0.1  
        if np.random.rand() < epsilon:  
            action = np.random.randint(0, output_dim)
        else:  
            action = torch.argmax(q_values).item()
        
        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        target = torch.tensor(reward + gamma * torch.max(agent(next_state)).item(), dtype=torch.float32)  # Q-target
        
        loss = criterion(q_values[action], target)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        
        state = next_state  
    

    if episode % 100 == 0:
        print(f"Episode {episode}, Loss: {loss.item()}")


torch.save(agent.state_dict(), "traffic_optimizer_model.pth")

