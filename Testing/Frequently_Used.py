import torch
import torch.nn as nn
import pickle
import json
import os
cwd = os.getcwd()
# Load user_to_index and item_to_index from pickle files
# Dictionary of user and item 
def index_transformer():
    user_to_index_path = os.path.join(cwd, 'pkl/user_to_index.pkl')
    item_to_index_path = os.path.join(cwd, 'pkl/item_to_index.pkl')
    index_to_item_path = os.path.join(cwd, 'pkl/index_to_item.pkl')
    with open(user_to_index_path, 'rb') as f:
        user_to_index = pickle.load(f)
    with open(item_to_index_path, 'rb') as f:
        item_to_index = pickle.load(f)
    with open(index_to_item_path, 'rb') as f:
        index_to_item = pickle.load(f)
    return user_to_index,item_to_index, index_to_item

def json_transform(address):    
    data = []
    with open(address, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class Content_Based_Model(nn.Module):
    def __init__(self, num_items, embedding_size):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size, 5192)
        self.fc2 = nn.Linear(5192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, embedding_size)  # Changed output size to 50

    def forward(self, item_indices):
        x = self.item_embedding(item_indices)
        mask = x != 0
        x = torch.masked_select(x, mask)
        padding = (self.fc1.in_features - x.size(0) % self.fc1.in_features) % self.fc1.in_features
        x = torch.cat([x, torch.zeros(padding)], dim=0).view(-1, self.fc1.in_features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class User_Based_Collab_Model(nn.Module):
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
        user_embedding = self.user_embedding(users).repeat(len(items), 1)
        item_embedding = self.item_embedding(items)
        x = torch.cat([user_embedding, item_embedding], dim=1)

        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()


