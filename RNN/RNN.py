# import package
import numpy as np
from sklearn.metrics import mean_absolute_error
import input_data_to_system as input_data
import torch
import torch.nn as nn
import json,os
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"

filename = 'processed_Industrial_and_Scientific.json'
preset_dataset = input_data.json_to_system(filename)
dataset = input_data.item_based_dataset(preset_dataset.__set_data__())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden = self.init_hidden()
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = self.fc(out)  
        return out

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.hidden_dim, dtype=torch.float32), 
                torch.zeros(self.num_layers, self.hidden_dim, dtype=torch.float32), 
                torch.zeros(self.num_layers, self.hidden_dim, dtype=torch.float32))


# train the model
model = LSTM(7, 50, 1,3)

criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
for epoch in range(50):
    
    #train loop
    test_mae = []
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        difference = np.abs((outputs.detach().numpy() - y.detach().numpy()))
        percentages = difference/y.detach().numpy()
        test_mae.append(np.mean(percentages))
    change = np.array(test_mae).mean()
    accuracy = (1-change) * 100
    
    

    print('Epoch {}: Loss {:.4f}'.format(epoch+1, loss.item()))
    print("Accuracy:", accuracy, "%")
    torch.save(model.state_dict(), f'model_{epoch}.pth')
