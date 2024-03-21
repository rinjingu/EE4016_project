
# import package
import input_data_to_system as input_data
import torch
import torch.nn as nn
import json,os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


filename = 'processed_Industrial_and_Scientific.json'
preset_dataset = input_data.json_to_system(filename)
dataset = input_data.item_based_dataset(preset_dataset.__set_data__())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class RNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden = self.init_hidden()
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out

    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim, dtype=torch.float32)

# train the model
model = RNN(7, 50, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    print('Epoch {}: Loss {:.4f}'.format(epoch+1, loss.item()))
