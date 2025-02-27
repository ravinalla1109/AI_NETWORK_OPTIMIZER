import torch
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

