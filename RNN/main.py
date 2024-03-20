import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class RecommenderRNN(nn.Module):
    def __init__(self, item_input_size, hidden_size, output_size):
        super(RecommenderRNN, self).__init__()

        self.item_embedding = nn.Embedding(item_input_size, hidden_size)

        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, item_input):
        item_embedded = self.item_embedding(item_input)

        _, output = self.rnn(item_embedded)

        output = self.fc(output[:, -1, :])

        return output

# Preprocessing and data loading
def preprocess_data(user_data_path, item_data_path):
    # Load the user base data
    user_data = pd.read_csv(user_data_path)
    # user_data = user_data[['user_id', 'product_id', 'rating']].values
    user_data = user_data[['user_id', 'product_id', 'rating']]  # Select specific columns
    user_data['user_id'] = user_data['user_id'].astype(str)
    user_data['product_id'] = user_data['product_id'].astype(str)
    user_data['rating'] = user_data['rating'].astype(int)
    user_data = pd.DataFrame(user_data)

    # Sort the data by review time if available
    # if 'review_time' in user_data.columns:
    #     user_data['review_time'] = pd.to_datetime(user_data['review_time'])
    #     user_data = user_data.sort_values('review_time')
    # print("user_data:", user_data)

    # Load the item base data
    item_data = pd.read_csv(item_data_path)
    # item_data = item_data[['product_id', 'type_of_product', 'category', 'brand', 'price']].values
    item_data = item_data[['product_id', 'type_of_product', 'category', 'brand', 'price']]
    item_data['product_id'] = item_data['product_id'].astype(str)
    item_data['type_of_product'] = item_data['type_of_product'].astype(str)
    item_data['category'] = item_data['category'].astype(str)
    item_data['brand'] = item_data['brand'].astype(str)
    item_data['price'] = item_data['price'].astype(float)
    item_data = pd.DataFrame(item_data)
    # print("item_data:", item_data)

    # Merge the user and item data based on the product ID
    merged_data = pd.merge(user_data, item_data, on='product_id')
    # print("merged_data:", merged_data)

    return merged_data

# Dataset class
class RecommenderDataset(Dataset):
    def __init__(self, data):
        self.user_indices = torch.tensor(data['user_index'].values, dtype=torch.long)
        self.item_indices = torch.tensor(data['item_index'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.user_indices)
    
    def __getitem__(self, index):
        return self.user_indices[index], self.item_indices[index], self.ratings[index]

# Recommendation function
def recommend_products(model, user_preferences, merged_data, item_id_to_index, top_n=5):
    user_index = len(item_id_to_index)  # Use a new index for the user
    user_indices = torch.tensor([user_index] * len(item_id_to_index), dtype=torch.long)
    item_indices = torch.tensor(list(item_id_to_index.values()), dtype=torch.long)

    with torch.no_grad():
        scores = model(user_indices, item_indices)

    _, indices = torch.topk(scores, top_n)
    top_product_ids = [list(item_id_to_index.keys())[i] for i in indices]

    recommended_products = merged_data[merged_data['product_id'].isin(top_product_ids)]
    
    return recommended_products

# Set the file paths for user data and item data
user_data_path = 'user_base_data.csv'
item_data_path = 'item_base_data.csv'

# Preprocess the data
merged_data = preprocess_data(user_data_path, item_data_path)

# Create a dictionary to map unique user and item IDs to indices
user_id_to_index = {user_id: i for i, user_id in enumerate(merged_data['user_id'].unique())}
item_id_to_index = {item_id: i for i, item_id in enumerate(merged_data['product_id'].unique())}

# Update the merged_data with the mapped indices
merged_data['user_index'] = merged_data['user_id'].map(user_id_to_index)
merged_data['item_index'] = merged_data['product_id'].map(item_id_to_index)

# Set the number of users, items, embedding dimension, and hidden size
num_users = len(user_id_to_index)
num_items = len(item_id_to_index)
embedding_dim = 32
hidden_size = 32

# Create an instance of the RNN model
# model = RecommenderRNN(num_users, num_items, embedding_dim, hidden_size)
model = RecommenderRNN(num_users, embedding_dim, hidden_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create the dataset and data loader
# print("merged_data:", merged_data)
dataset = RecommenderDataset(merged_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for user_indices, item_indices, ratings in data_loader:
        optimizer.zero_grad()

        # outputs = model(user_indices, item_indices)
        outputs = model(user_indices)
        loss = criterion(outputs, ratings.unsqueeze(1))
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(data_loader)}")

# User preference input
user_preferences = {
    'type_of_product': 'Electronics',
    'category': 'Laptops',
    'brand': 'Apple',
    'price': 2000.0
}

# Recommendation
recommended_products = recommend_products(model, user_preferences, merged_data, item_id_to_index)

print("Recommended Products:")
print(recommended_products[['product_id', 'type_of_product', 'category', 'brand', 'price']])
