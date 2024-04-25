import torch
from torch import nn
import pickle

# Load user_to_index and business_to_index from pickle files
# Dictionary of user and business 
def index_transformer():
    with open('pkl/user_to_index.pkl', 'rb') as f:
        user_to_index = pickle.load(f)
    with open('pkl/business_to_index.pkl', 'rb') as f:
        business_to_index = pickle.load(f)
    with open('pkl/index_to_business.pkl', 'rb') as f:
        index_to_business = pickle.load(f)
    return user_to_index,business_to_index, index_to_business



user_to_index,business_to_index,index_to_business = index_transformer()
  
class UserBusinessModel(nn.Module):
    def __init__(self, num_users, num_businesses, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.business_embedding = nn.Embedding(num_businesses, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, users, businesses):
        user_embedding = self.user_embedding(users)
        business_embedding = self.business_embedding(businesses)
 
        x = torch.cat([user_embedding, business_embedding], dim=1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()

# Load the model
num_users = len(user_to_index)
num_businesses = len(business_to_index)
embedding_size = 80
model = UserBusinessModel(num_users, num_businesses, embedding_size)
model.load_state_dict(torch.load('trained_models/collab_test1.pth'))
max_user_index = max(user_to_index.values())
max_business_index = max(business_to_index.values())
print(f'Max user index: {max_user_index}, Max business index: {max_business_index}')
print(f'User embedding size: {model.user_embedding.weight.size(0)}, Business embedding size: {model.business_embedding.weight.size(0)}')

# User interaction loop
while True:
    mode = input("Enter 'user' to recommend businesses for a user, 'business' to predict the rating for a business, or 'quit' to exit: ")
    if mode.lower() == 'quit':
        break
    elif mode.lower() == 'user':
        user_id = input("Enter user id: ")
        if user_id in user_to_index and user_to_index[user_id] != -1 and user_to_index[user_id] < model.user_embedding.weight.size(0):
            user_index = torch.tensor([user_to_index[user_id]] * len(business_to_index), dtype=torch.long)
            business_indices = torch.tensor(list(business_to_index.values()), dtype=torch.long)
            ratings = model(user_index, business_indices)
            ratings = torch.sigmoid(ratings) * 4 + 1  # Convert the ratings to a range of 1-5
            top10_values, top10_indices = torch.topk(ratings, 10)
            top10_businesses = [index_to_business[index.item()] for index in top10_indices.detach()]
            top10_ratings = top10_values.detach().cpu().numpy()
            print(f'Top 10 recommended businesses for user {user_id} and their predicted ratings:')
            for business, rating in zip(top10_businesses, top10_ratings):
                print(f'Business: {business}, Predicted rating: {rating:.2f}')
        else:
            print("User id not found or out of range. Please try again.")

    elif mode.lower() == 'business':
        business_id = input("Enter business id: ")
        if business_id in business_to_index and business_to_index[business_id] < model.business_embedding.weight.size(0):
            business_index = torch.tensor([business_to_index[business_id]], dtype=torch.long)
            user_indices = torch.tensor([index for index in user_to_index.values() if index != -1 and index < model.user_embedding.weight.size(0)], dtype=torch.long)
            business_index = business_index.repeat(user_indices.size(0))
            ratings = model(user_indices, business_index)
            ratings = torch.sigmoid(ratings) * 4 + 1  # Convert the ratings to a range of 1-5
            print(f'Predicted rating for business {business_id}: {ratings.mean().item():.2f}')
        else:
            print("Business id not found or out of range. Please try again.")
    else:
        print("Invalid mode. Please enter 'user' or 'business'.")