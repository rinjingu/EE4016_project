# import package
import torch
import torch.nn as nn
import json,os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from tqdm import tqdm

cwd = os.getcwd() # Set current working directory
# Load JSON data from a file
preset_data = [] #add read size
with open(os.path.join(cwd, 'processed_Industrial_and_Scientific.json'), 'r') as f:
    for line in f:
        preset_data.append(json.loads(line))

class item_based_dataset(Dataset):
    def __init__(self, preset_data):
        self.preset_data = preset_data

    def __getitem__(self, index):
        try:
            asin = self.preset_data[index]['asin']
            price_from = float(self.preset_data[index]['price'][0]) or -1
            price_to = float(self.preset_data[index]['price'][1]) or -1
            avg_rating = self.preset_data[index]['avg_rating'] or -1
            rank = self.preset_data[index]['rank'] or -1
            brand = self.preset_data[index]['brand'] or -1
            category = self.preset_data[index]['category'] or [-1, -1]
            category_a, category_b = (category + [-1, -1])[:2]
            activeness = self.preset_data[index]['activeness'] or -1
            packed_data = [price_from,price_to,rank,brand,category_a,category_b,activeness]
            trainset = torch.tensor(packed_data)  # convert to tensor
            Verify_ans = torch.tensor([avg_rating], dtype=torch.float)
            return trainset,Verify_ans
        except KeyError as e:
            print(f"KeyError: {e} is not found in the data at index {index}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __len__(self):
        return len(self.preset_data)

class ItemRecommenderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ItemRecommenderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.unsqueeze(2) ### critical, no sure if it is okay, it is now runnable but seems not useful
        # print(x.shape)
        # print(h0.shape)
        # Forward propagate through the RNN layer
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out


# Initialize the data loader
dataset = item_based_dataset(preset_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


input_size = 1  # Number of input features
num_layers = 5  # Specify the number of stacked RNN layers
hidden_size = 64  # Size of the hidden state in the RNN
output_size = 1  # Number of output units (rating prediction)
model = ItemRecommenderRNN(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model using the DataLoader
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for features, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


"""
# Assuming you have trained your model and have prepared input data
input_data = ...  # Prepare your input data

# Pass the input data through the trained model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(input_data)

# Generate recommendations based on the model's output
# Recommendation generation logic depends on your specific system design

# For example, you can recommend the top-k items with the highest predicted ratings
k = 10  # Number of recommendations to generate
top_k_items = torch.topk(output, k).indices

# You can then use the recommended item indices to retrieve the corresponding item information
recommended_items = item_data[top_k_items]

# Print the recommended items
print("Recommended Items:")
for item in recommended_items:
    print(item)

# Use the recommended items for further processing or display to the user
"""

###############################
# # User preference input
# user_preferences = {
#     'type_of_product': 'Electronics',
#     'category': 'Laptops',
#     'brand': 'Apple',
#     'price': 2000.0
# }

# # Recommendation
# recommended_products = recommend_products(model, user_preferences, merged_data, item_id_to_index)

# print("Recommended Products:")
# print(recommended_products[['product_id', 'type_of_product', 'category', 'brand', 'price']])
###############################
    