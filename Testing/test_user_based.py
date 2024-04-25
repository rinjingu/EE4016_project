import torch
from torch import nn
import pickle

# Load user_to_index and item_to_index from pickle files
# Dictionary of user and item 
def index_transformer():
    with open('pkl/user_to_index.pkl', 'rb') as f:
        user_to_index = pickle.load(f)
    with open('pkl/item_to_index.pkl', 'rb') as f:
        item_to_index = pickle.load(f)
    with open('pkl/index_to_item.pkl', 'rb') as f:
        index_to_item = pickle.load(f)
    return user_to_index,item_to_index, index_to_item



user_to_index,item_to_index,index_to_item = index_transformer()
  
class UserCollabModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
 
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()

# Load the model
num_users = len(user_to_index)
num_items = len(item_to_index)
embedding_size = 80
model = UserCollabModel(num_users, num_items, embedding_size)
model.load_state_dict(torch.load('trained_models/collab_test1.pth'))
max_user_index = max(user_to_index.values())
max_item_index = max(item_to_index.values())
print(f'Max user index: {max_user_index}, Max item index: {max_item_index}')
print(f'User embedding size: {model.user_embedding.weight.size(0)}, item embedding size: {model.item_embedding.weight.size(0)}')

# User interaction loop
while True:
    mode = input("Enter 'user' to recommend items for a user, 'item' to predict the rating for a item, or 'quit' to exit: ")
    if mode.lower() == 'quit':
        break
    elif mode.lower() == 'user':
        user_id = input("Enter user id: ")
        if user_id in user_to_index and user_to_index[user_id] != -1 and user_to_index[user_id] < model.user_embedding.weight.size(0):
            user_index = torch.tensor([user_to_index[user_id]] * len(item_to_index), dtype=torch.long)
            item_indices = torch.tensor(list(item_to_index.values()), dtype=torch.long)
            ratings = model(user_index, item_indices)
            ratings = torch.sigmoid(ratings) * 4 + 1  # Convert the ratings to a range of 1-5
            top10_values, top10_indices = torch.topk(ratings, 10)
            top10_items = [index_to_item[index.item()] for index in top10_indices.detach()]
            top10_ratings = top10_values.detach().cpu().numpy()
            print(f'Top 10 recommended items for user {user_id} and their predicted ratings:')
            for item, rating in zip(top10_items, top10_ratings):
                print(f'item: {item}, Predicted rating: {rating:.2f}')
        else:
            print("User id not found or out of range. Please try again.")

    elif mode.lower() == 'item':
        item_id = input("Enter item id: ")
        if item_id in item_to_index and item_to_index[item_id] < model.item_embedding.weight.size(0):
            item_index = torch.tensor([item_to_index[item_id]], dtype=torch.long)
            user_indices = torch.tensor([index for index in user_to_index.values() if index != -1 and index < model.user_embedding.weight.size(0)], dtype=torch.long)
            item_index = item_index.repeat(user_indices.size(0))
            ratings = model(user_indices, item_index)
            ratings = torch.sigmoid(ratings) * 4 + 1  # Convert the ratings to a range of 1-5
            print(f'Predicted rating for item {item_id}: {ratings.mean().item():.2f}')
        else:
            print("item id not found or out of range. Please try again.")
    else:
        print("Invalid mode. Please enter 'user' or 'item'.")