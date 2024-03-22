import input_data_to_system as input_data
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import json
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

# Instantiate the dataset and dataloader
filename = 'processed_Industrial_and_Scientific.json'
preset_dataset = input_data.json_to_system(filename)
dataset = input_data.item_based_dataset(preset_dataset.__set_data__())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        x = self.fc4(x)
        return x


# Define hyperparameters
input_size = 7  # Dimension of input features
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # Dimension of output (predicted rating)

# Instantiate the model
model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)




num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    test_mae = []
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)  # Multiply by batch size
        difference = np.abs((outputs.detach().numpy() - targets.detach().numpy()))
        test_mae.append(np.mean(difference))
    accuracy = np.array(test_mae).mean()
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
    print("Accuracy:", accuracy, "%")
    torch.save(model.state_dict(), f'model_{epoch}.pth')
