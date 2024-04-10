from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import pickle
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Load user_to_index and business_to_index from pickle files
with open('pkl/user_to_index.pkl', 'rb') as f:
    user_to_index = pickle.load(f)
with open('pkl/business_to_index.pkl', 'rb') as f:
    business_to_index = pickle.load(f)
with open('pkl/index_to_business.pkl', 'rb') as f:
    index_to_business = pickle.load(f)

# Get a list of all business indices
all_business_indices = list(business_to_index.values())

# Convert to a tensor
all_business_indices = torch.tensor(all_business_indices)
import json

data = []
with open('yelp/process_user.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Split the data into a training set and a test set
train_set, test_set = train_test_split(data, test_size=0.0001, random_state=42)


def precision_recall_at_k(recommended_items, test_items, k):
    # Get the top-k items from the recommended items
    top_k_items = recommended_items[:k]

    # Calculate the number of relevant items (items in both recommended and test set)
    relevant_items = len(set(top_k_items) & set(test_items))

    # Calculate precision and recall
    precision = relevant_items / len(top_k_items) if top_k_items else 0
    recall = relevant_items / len(test_items) if test_items else 0

    return precision, recall


class ContentModel(nn.Module):
    def __init__(self, num_items, embedding_size):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size, 5192)
        self.fc2 = nn.Linear(5192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 1)  # Changed output size to 50

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
        user_embedding = self.user_embedding(users).repeat(len(businesses), 1)
        business_embedding = self.business_embedding(businesses)
        x = torch.cat([user_embedding, business_embedding], dim=1)

        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()

class HybridModel(nn.Module):
    def __init__(self, content_based_model, collaborative_model, alpha=0.5):
        super().__init__()
        self.content_based_model = content_based_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha

    def forward(self, users, items):
        content_based_output = self.content_based_model(items)
        collaborative_output = self.collaborative_model(users, items)
        print(content_based_output)
        print(collaborative_output)
        return self.alpha * content_based_output.squeeze() + (1 - self.alpha) * collaborative_output
# Load the models
embedding_size = 50
content_based_model = ContentModel(len(business_to_index), embedding_size)
content_based_model.load_state_dict(torch.load('trained_model/content-based1.pth'))
collaborative_model = UserBusinessModel(len(user_to_index), len(business_to_index), 80)
collaborative_model.load_state_dict(torch.load('trained_model/colab_user-based_model.pth'))

# For HybridModel
hybrid_model = HybridModel(content_based_model, collaborative_model, alpha=0.5)

# Get user_id from user input
user_id = input("Please enter your user ID: ")

# Convert user_id to index
try:
    user_index = user_to_index[user_id]
except KeyError:
    print("Invalid user ID. Please enter a valid user ID.")
    exit()

user_index_tensor = torch.tensor([user_index])
output = hybrid_model(user_index_tensor, all_business_indices)

# Get the indices of the businesses with the highest scores
top_business_indices = output.argsort(descending=True)[:10]  # Change 10 to the number of recommendations you want

# Convert the indices back to business IDs
recommended_businesses = [index_to_business[index.item()] for index in top_business_indices]

print("Recommended businesses for user {}: {}".format(user_id, recommended_businesses))

# Calculate average precision and recall
avg_precision = 0
avg_recall = 0
k = 10  # Number of recommendations to consider

for user_ratings in tqdm(test_set):
    user_id = user_ratings["user_id"]
    test_items = {k: v for k, v in user_ratings.items() if k != "user_id"}

    # Convert user_id to index
    user_index = user_to_index[user_id]
    user_index_tensor = torch.tensor([user_index])

    # Get recommended items from the model
    output = hybrid_model(user_index_tensor, all_business_indices)
    recommended_indices = output.argsort(descending=True)
    recommended_items = [index_to_business[index.item()] for index in recommended_indices]

    # Calculate precision and recall
    precision, recall = precision_recall_at_k(recommended_items, test_items, k)

    # Update average precision and recall
    avg_precision += precision
    avg_recall += recall

# Calculate average precision and recall
avg_precision /= len(test_set)
avg_recall /= len(test_set)

print("Average Precision@{}: {}".format(k, avg_precision))
print("Average Recall@{}: {}".format(k, avg_recall))