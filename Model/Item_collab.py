import os
import pickle
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import Frequently_Used as fu
from annoy import AnnoyIndex
import torch.nn.functional as F



has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
cwd = os.getcwd()



class NDCGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def DCG(self, items):
        return torch.sum(items / (torch.log2(torch.arange(2, items.size(0) + 2).float().to(items.device))))

    def forward(self, predictions, targets):
            # Ensure predictions and targets are not empty and have at least one dimension
        if predictions.nelement() == 0 or targets.nelement() == 0:
            print("Error: predictions or targets is empty.")
            return
        if len(predictions.shape) == 0 or len(targets.shape) == 0:
            print("Error: predictions or targets doesn't have at least one dimension.")
            return
        # Ensure predictions and targets are the same shape
        assert predictions.shape[0] == targets.shape[0]
        # Calculate DCG
        dcg = self.DCG(predictions)
        # Calculate IDCG
        idcg = self.DCG(targets.sort(descending=True)[0])
        # Calculate NDCG
        ndcg = dcg / idcg if idcg != 0  else 0
        # Return loss as 1 - NDCG
        loss = abs(1 - ndcg)
        return loss

class UserItemDataset(Dataset):
    def __init__(self, user_item_pairs,item_to_index):
        self.user_item_pairs = user_item_pairs
        self.item_to_index = item_to_index
    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item, rating = self.user_item_pairs[idx]
        item = self.item_to_index[item]
        return user, item, rating
    
class RecommendationModel(nn.Module):
    def __init__(self, user_histories, embeeded_size, num_items):
        super().__init__()
        self.user_histories = user_histories
        self.item_embedding = nn.Embedding(num_items, embeeded_size)  # Add this line
        self.linear1 = nn.Linear(embeeded_size+1, 2048)  # Increase the size of the first linear layer
        self.dropout1 = nn.Dropout(0.4)  # Add dropout for regularization
        self.linear2 = nn.Linear(2048, 1024)  # Add a second linear layer
        self.dropout2 = nn.Dropout(0.3)  # Add dropout for regularization
        self.linear3 = nn.Linear(1024, 256)  # Final linear layer to get a single score
        self.dropout3 = nn.Dropout(0.2)  # Add dropout for regularization
        self.linear4 = nn.Linear(256, 1)  # Final linear layer to get a single score


    def forward(self, similar_item,):
        # Get the item indices and distances
        item_indices, distances = similar_item[:, 0].long(), similar_item[:, 1]
        # Convert the indices to embeddings
        item_embeddings = self.item_embedding(item_indices)
        # Combine the embeddings and distances
        x = torch.cat([item_embeddings, distances.view(-1, 1)], dim=1)
        # Pass the embedding through the linear layers
        x = self.linear1(x)
        x = F.leaky_relu(x)  # Add a ReLU activation function
        x = self.dropout1(x)  # Apply dropout
        x = self.linear2(x)
        x = F.leaky_relu(x)  # Add a ReLU activation function
        x = self.dropout2(x)  # Apply dropout
        x = self.linear3(x)
        x = F.leaky_relu(x)  # Add a ReLU activation function
        x = self.dropout3(x)  # Apply dropout
        score = self.linear4(x)
        score = torch.sigmoid(score)


        return score.squeeze()  # Make sure to return a scalar

    def recommend(self, user_id, num_recommendations=10):
        user_history = self.user_histories[user_id]
        recommendation_scores = {}

        for item in user_history:
            # Get the most similar items
            similar_items = bussiness_index.get_nns_by_vector(self.item_embedding(torch.tensor([item_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)
            for similar_item, similarity_score in zip(*similar_items):
                if similar_item in recommendation_scores:
                    recommendation_scores[similar_item] += similarity_score
                else:
                    recommendation_scores[similar_item] = similarity_score

        # Sort the items by their total similarity score
        recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

        # Return the top num_recommendations items
        return [index_to_item[i[0]] for i in recommendations[:num_recommendations]]
    
def load_user_item_pairs(address):
    data = fu.json_transform(address)
    user_item_pairs = []
    for user_data in data:
        user_id = user_data['user_id']
        for item_id, rating in user_data.items():
            if item_id != 'user_id':
                user_item_pairs.append((user_id, item_id, rating))
    return user_item_pairs

# Load mappings
user_item_pairs = load_user_item_pairs(os.path.join(cwd, 'yelp/process_user.json'))

# Load Annoy index
bussiness_index = AnnoyIndex(128, 'angular')  # 50 is the dimensionality of your item embeddings
bussiness_index.load('yelp_item.ann')
# Create mappings
user_to_index, item_to_index, index_to_item = fu.index_transformer()

# Convert the list to a DataFrame
user_item_pairs_df = pd.DataFrame(user_item_pairs, columns=['user_id', 'item_id', 'rating'])

# Now you can call to_records on the DataFrame
user_item_pairs_list = [tuple(x) for x in user_item_pairs_df.to_records(index=False)]
user_to_index = {user: i for i, user in enumerate(set(user_id for user_id, item_id, rating in user_item_pairs_list))}
item_to_index = {item: i for i, item in enumerate(set(item_id for user_id, item_id, rating in user_item_pairs_list))}

# Load your pretrained model
# Load data
dataset = UserItemDataset(user_item_pairs,item_to_index)
data_loader = DataLoader(dataset, batch_size=128)
user_item_pairs = pd.DataFrame(user_item_pairs, columns=['user_id', 'item_id', 'rating'])
with open('user_item_pairs.pkl', 'wb') as f:
    pickle.dump(user_item_pairs, f)
    
user_buying_records = [(user_to_index[row.user_id], item_to_index[row.item_id], row.rating) for row in user_item_pairs.itertuples()]
user_buying_records_tensor = torch.tensor([(x[0], x[1], x[2]) for x in user_buying_records])
embeeded_size = 128
# Create model, loss function, and optimizer
user_histories_file = fu.json_transform(os.path.join(cwd, 'yelp/process_user.json'))

model = RecommendationModel(user_histories_file,embeeded_size,len(item_to_index))
loss_function = NDCGLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
# Train the model
for epoch in range(5):
    model.train()
    train_loss = 0
    for user, item, rating in tqdm(data_loader):
        # Get the most similar items
        if item.size(0) != embeeded_size:
            item = item.resize_(embeeded_size)
        similar_items = bussiness_index.get_nns_by_vector(item, data_loader.batch_size, search_k=-1, include_distances=True)
        # Get the most similar items and their distances
        similar_item_indices, distances = similar_items
        # Convert the lists to tensors
        similar_item_indices = torch.tensor(similar_item_indices)
        distances = torch.tensor(distances)
        # Reshape the tensors to 2D
        similar_item_indices = similar_item_indices.view(-1, 1)
        distances = distances.view(-1, 1)
        # Concatenate the tensors along the second dimension
        similar_items_tensor = torch.cat([similar_item_indices, distances], dim=1)
        rating = (rating.float()-1)/4
        # Predict the user's interaction with these similar items
        predictions = model(similar_items_tensor)

        predictions = predictions[:rating.size(0)]
        # Ensure that the rating tensor has the same shape as the predictions tensor
        rating = rating * torch.ones_like(predictions)

        # Compute the loss
        loss = loss_function(predictions, rating)

        # Backward pass
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimization step
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(data_loader)}')
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
    