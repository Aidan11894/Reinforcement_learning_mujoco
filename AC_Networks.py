import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, chkpt_dir="tmp/ddpg", hidden_layers_array=None):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        self.hidden_layers_array = hidden_layers_array

        if hidden_layers_array is None:
            hidden_layers_array = [256, 256]

        # --------------------Finished uo to here properly. ------------------------------------

        # Input dimension is embedding_dim (from state) plus action dimensions
        layers = []
        x = self.input_dims
        for h_dim in hidden_layers_array:
            layers.append(nn.Linear(x, h_dim))
            layers.append(nn.ReLU())
            x = h_dim
        layers.append(nn.Linear(self.input_dims, 1))  # Output scalar Q-value
        self.model = nn.Sequential(*layers)
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # Add batch dimension if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Shape becomes [1, state_dim]
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Shape becomes [1, action_dim]

        x = T.cat([state, action], dim=1)
        return self.model(x)



