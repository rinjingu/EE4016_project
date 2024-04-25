import os
import pickle
from annoy import AnnoyIndex
import torch
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import Frequently_Used as fu
from torch.utils.data.dataset import random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import json

has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "cpu"

cwd = os.getcwd()
bussiness_index = AnnoyIndex(128, 'angular') 
bussiness_index.load('ann/yelp_business.ann')
# Load the mappings
with open('pkl/user_to_index.pkl', 'rb') as f: 
    user_to_index = pickle.load(f)
with open('pkl/business_to_index.pkl', 'rb') as f:
    business_to_index = pickle.load(f)
with open('pkl/index_to_business.pkl', 'rb') as f:
    index_to_business = pickle.load(f)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

class UserItemDataset(Dataset):
    def __init__(self, user_item_pairs,business_to_index):
        self.user_item_pairs = user_item_pairs
        self.business_to_index = business_to_index
    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item, rating = self.user_item_pairs[idx]
        item = self.business_to_index[item]
        return user, item, rating
    

class NDCGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def DCG(self, items):
        items = torch.from_numpy(items).float().to(self.device)
        return torch.sum(items / (torch.log2(torch.arange(2, items.size()[0] + 2).to(self.device))))

    def forward(self, predictions, targets):
        # Ensure predictions and targets are not empty and have at least one dimension
        if predictions.size == 0 or targets.size == 0:
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
        idcg = self.DCG(np.sort(targets)[::-1].copy())
        # Calculate NDCG
        ndcg = dcg / idcg if idcg != 0  else 0
        # Return loss as 1 - NDCG
        loss = abs(1 - ndcg)
        return loss


class ReviewDataset(Dataset):
    def __init__(self, df):
        self.reviews = df['text'].tolist()
        self.labels = df['stars'].tolist()
        self.business_ids = df['business_id'].tolist()

    def __getitem__(self, idx):
        review = self.reviews[idx]
        business_id = self.business_ids[idx]
        label = self.labels[idx]
        inputs = tokenizer.encode_plus(
            f"{business_id} {review}",
            None,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label - 1, dtype=torch.long)
        }
    def __len__(self):
        return len(self.reviews)

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
    
    def recommend(self, attention_weight, user_id, num_recommendations=20):
        user_history = self.user_histories[user_id]
        recommendation_scores = {}
        for item in user_history:
            # Get the most similar items
            if item == 'user_id':
                continue
            similar_items = bussiness_index.get_nns_by_vector(self.item_embedding(torch.tensor([business_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)           
            for similar_item, similarity_score in zip(*similar_items):
                similarity_score *= attention_weight
                if(similarity_score > 5):
                    similarity_score = 5
                if similar_item in recommendation_scores:
                    recommendation_scores[similar_item] += similarity_score
                else:
                    recommendation_scores[similar_item] = similarity_score
        recommendation_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendation_scores[:num_recommendations]


    def get_score(self, business_index, distance):
        business_embeddings = self.business_embedding(business_index).to(device)
        x = torch.cat([business_embeddings, distance.view(-1, 1).to(device)], dim=1)
        self.fc1 = nn.Linear(81, 1024)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()


class RecommendationModel(nn.Module):
    def __init__(self, user_histories, embedded_size, num_items):
        super().__init__()
        self.user_histories = user_histories
        self.item_embedding = nn.Embedding(num_items, embedded_size)
        self.linear1 = nn.Linear(embedded_size+1, 2048)
        self.dropout1 = nn.Dropout(0.4)
        self.linear2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(1024, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, similar_item):
        item_indices, distances = similar_item[:, 0].long(), similar_item[:, 1]
        item_embeddings = self.item_embedding(item_indices)
        x = torch.cat([item_embeddings, distances.view(-1, 1)], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout3(x)
        return self.linear4(x).squeeze()

    def recommend(self, attention_weight, user_id, num_recommendations=20):
        user_history = self.user_histories[user_id]
        recommendation_scores = {}
        for item in user_history:
            # Get the most similar items
            if item == 'user_id':
                continue
            similar_items = bussiness_index.get_nns_by_vector(self.item_embedding(torch.tensor([business_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)           
            for similar_item, similarity_score in zip(*similar_items):
                similarity_score *= attention_weight
                if similar_item in recommendation_scores:
                    recommendation_scores[similar_item] += similarity_score
                else:
                    recommendation_scores[similar_item] = similarity_score
        recommendation_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendation_scores[:num_recommendations]

    def get_score(self, item_index, distance):
        item_embeddings = self.item_embedding(item_index).to(device)
        x = torch.cat([item_embeddings, distance.view(-1, 1).to(device)], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout3(x)
        score = torch.sigmoid(self.linear4(x))
        return score.squeeze()

class BusinessModel(nn.Module):
    def __init__(self, transformer_model, recommendation_model, df):
        super().__init__()
        self.transformer_model = transformer_model
        self.recommendation_model = recommendation_model
        self.df = df
    
    def forward(self, business_id):
        # Use the transformer model to predict the rating
        rating = predict_rating(self.transformer_model,business_id,self.df)
        rating = torch.tensor(rating)
        # Normalize the rating to get the attention weight
        attention_weight = F.softmax(rating, dim=-1)
        # Use the attention weight as the distance in the recommendation model
        score = self.recommendation_model.get_score(business_id, attention_weight)
        return score

    def getscore(self,business_id, attention_weight):
        score = self.recommendation_model.get_score(business_id, attention_weight)
        return score
    
    def recommend(self, user_id, num_recommendations=20):
        user_history = self.recommendation_model.user_histories[user_id]
        recommendation_scores = {}
        
        for item in user_history:
            if item == 'user_id':
                continue
            # Use the transformer model to predict the rating
            rating = predict_rating(self.transformer_model,item,self.df)
            # Normalize the rating to get the attention weight
            rating = torch.tensor(rating)  # Convert the list to a tensor
            attention_weight = rating / 5  # Normalize the rating
            # Use the attention weight as the distance in the recommendation model
            similar_items = self.recommendation_model.recommend(attention_weight,user_id, num_recommendations)
            for similar_item, similarity_score in similar_items:
                similar_item = index_to_business[similar_item]
                similarity_score *= rating
                if similarity_score >5: #up
                    similarity_score = torch.tensor(5)
                similarity_score /= (len(user_history)-1)

                # Use dict.get() to avoid KeyError
                recommendation_scores[similar_item] = recommendation_scores.get(similar_item, 0) + similarity_score.item()  # Convert tensor to a Python number

        recommendation_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

        return recommendation_scores[:num_recommendations]
    
class UserModel(nn.Module):
    def __init__(self, transformer_model, user_model, df):
        super().__init__()
        self.transformer_model = transformer_model
        self.user_model = user_model
        self.df = df

    def forward(self, business_id):
        # Use the transformer model to predict the rating
        rating = predict_rating(self.transformer_model,business_id,self.df)
        rating = torch.tensor(rating)
        # Normalize the rating to get the attention weight
        attention_weight = F.softmax(rating, dim=-1)
        # Use the attention weight as the distance in the recommendation model
        score = self.user_model.get_score(business_id, attention_weight)
        return score

    def getscore(self,user_id,business_id, attention_weight):
        score = self.user_model.get_score(business_id, attention_weight)
        return score
    
    def recommend(self, user_id,user_histories_file, num_recommendations=50):
        user_history = next((history for history in user_histories_file if history['user_id'] == user_id), None)
        if user_history is None:
            print(f'User {user_id} not found in the history.')
            return []

        if user_id in user_to_index and user_to_index[user_id] != -1 and user_to_index[user_id] < self.user_model.user_embedding.weight.size(0):
            user_index = torch.tensor([user_to_index[user_id]] * len(business_to_index), dtype=torch.long)
            business_indices = torch.tensor(list(business_to_index.values()), dtype=torch.long)
            rating = self.user_model(user_index, business_indices)
            rating = torch.sigmoid(rating) * 4 + 1  # Convert the ratings to a range of 1-5
            # Use the attention weight as the distance in the recommendation model
            
            top20_values, top20_indices = torch.topk(rating, 20)
            top20_recommendations = {}
            for index, value in zip(top20_indices.detach(), top20_values.detach().cpu().numpy()):
                attention_weights = predict_rating(self.transformer_model, index_to_business[index.item()], self.df)
                attention_weight = np.mean(attention_weights)
                attention_weight = attention_weight / 5
                
                top20_recommendations[index_to_business[index.item()]] = value * attention_weight

            top20_recommendations = sorted(top20_recommendations.items(), key=lambda x: x[1], reverse=True)
            return top20_recommendations

def load_user_item_pairs(address):
    data = fu.json_transform(address)
    user_item_pairs = []
    for user_data in data:
        user_id = user_data['user_id']
        for business_id, rating in user_data.items():
            if business_id != 'user_id':
                user_item_pairs.append((user_id, business_id, rating))
    return user_item_pairs

def get_reviews_for_business(business_id, df):
    reviews = df[df['business_id'] == business_id]['text'].tolist()
    return reviews

def predict_rating(model, business_id, df):
    attention_weights = []
    comments = get_reviews_for_business(business_id, df)
    df1 = pd.DataFrame({
        'text': comments,
        'business_id': [business_id] * len(comments),
        'stars': [0] * len(comments)  # Dummy labels
    })
    dataset = ReviewDataset(df1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    all_predictions = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1) + 1
        all_predictions.extend(predictions.tolist())
    attention_weights.append((sum(all_predictions) / len(all_predictions)))
    return attention_weights


# Load the DataFrame from the JSON file
df = pd.read_json('yelp/processed_review.json',lines=True)
bussiness_data = pd.read_json('yelp/processed_business.json',lines=True)

# Load the trained model
transformer_model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=5)
transformer_model.load_state_dict(torch.load('trained_model/trained_model_20240422015913.pth'))
transformer_model.eval()
transformer_model.to(device)
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')


user_to_index, business_to_index, index_to_business = fu.index_transformer()
user_histories_file = fu.json_transform(os.path.join(cwd, 'yelp/process_user.json'))
embedded_size = 128
embedding_size = 80
item_based_model = RecommendationModel(user_histories_file, embedded_size, len(business_to_index))
item_based_model.load_state_dict(torch.load(os.path.join(cwd, 'trained_model/Item-based_model.pth')))
item_based_model = item_based_model.to(device)
business_model = BusinessModel(transformer_model, item_based_model,df)
business_model = business_model.to(device)

user_based_model = UserBusinessModel(len(user_to_index), len(business_to_index), embedding_size)
user_based_model.load_state_dict(torch.load(os.path.join(cwd, 'trained_model/collab_test1.pth')))
user_based_model = user_based_model.to(device)

user_model = UserModel(transformer_model, user_based_model,df)
user_model = user_model.to(device)

# Load mappings
user_item_pairs = load_user_item_pairs(os.path.join(cwd, 'yelp/process_user.json'))

# Convert the list to a DataFrame
user_item_pairs_df = pd.DataFrame(user_item_pairs, columns=['user_id', 'business_id', 'rating'])

# Now you can call to_records on the DataFrame
user_item_pairs_list = [tuple(x) for x in user_item_pairs_df.to_records(index=False)]
user_to_index = {user: i for i, user in enumerate(set(user_id for user_id, business_id, rating in user_item_pairs_list))}
business_to_index = {business: i for i, business in enumerate(set(business_id for user_id, business_id, rating in user_item_pairs_list))}

# Load your pretrained model
# Load data
dataset = UserItemDataset(user_item_pairs,business_to_index)
data_loader = DataLoader(dataset, batch_size=48)

user_id = '-L92vRdiCwz6QWjvOtS-zA'

# # Ensure the model is in evaluation mode
# user_model.eval()
# # Get recommendations for a user
# user_recommendations = user_model.recommend(user_id,user_histories_file)

# # Print the recommendations
# for item, score in user_recommendations:
#     print(f'Item: {item}, Score: {score}')


# 

# Ensure the model is in evaluation mode
# business_model.eval()
# # Get recommendations for a user
# business_recommendations = business_model.recommend(user_to_index[user_id])

# # Print the recommendations
# for item, score in business_recommendations:
    
#     print(f'Item: {item}, Score: {score}')
    
    
def get_business_score(user_id, business_id, user_model,business_model, transformer_model, df):
    # Use the transformer model to predict the rating
    rating = predict_rating(transformer_model, business_id, df)
    rating = torch.tensor(rating)
    # Convert business_id to business_index
    business_index = business_to_index[business_id]
    # Use the transformer model to predict the rating
    predicted_rating = predict_rating(transformer_model, business_id, df)
    predicted_rating = torch.tensor(predicted_rating)
    # Normalize the predicted rating to get the attention weight
    attention_weight = F.softmax(predicted_rating, dim=-1)
    business_index = torch.tensor([business_index]).to(device)
    # Use the attention weight as the distance in the recommendation model
    score = (user_model.getscore(user_id, business_index, attention_weight) + business_model.getscore(business_index, attention_weight)) / 2
    score = score.item()
    return score

# Randomly select 100 users
selected_users = random.sample([user_history['user_id'] for user_history in user_histories_file], min(100, len(user_histories_file)))

rmse_values = []
mae_values = []
ndcg_values = []

ndcg_loss_fn = NDCGLoss(device)

for user_id in tqdm(selected_users):
    # Get the user history for the selected user
    user_history = next((user_history for user_history in user_histories_file if user_history['user_id'] == user_id), None)
    if user_history is None:
        continue

    true_ratings_list = []
    predicted_ratings_list = []
    print(user_history)
    # For each business in the user history, predict the score and compare it with the actual rating
    for business_id, true_rating in user_history.items():
        if business_id == 'user_id':
            continue
        predicted_rating = get_business_score(user_id, business_id, user_model, business_model, transformer_model, df)
        true_ratings_list.append(true_rating)
        predicted_ratings_list.append(predicted_rating)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(true_ratings_list, predicted_ratings_list))
    rmse_values.append(rmse)
    mae = mean_absolute_error(true_ratings_list, predicted_ratings_list)
    mae_values.append(mae)
    ndcg_loss = ndcg_loss_fn(np.array([true_ratings_list]), np.array([predicted_ratings_list]))
    ndcg_values.append(ndcg_loss)

# Calculate average RMSE, MAE, and NDCG loss
average_rmse = np.mean(rmse_values)
average_mae = np.mean(mae_values)
average_ndcg_loss = np.mean(ndcg_values)

print(f'Average RMSE: {average_rmse}')
print(f'Average MAE: {average_mae}')
print(f'Average NDCG Loss: {average_ndcg_loss}')




# Randomly select 100 users
# selected_users = random.sample([user_history['user_id'] for user_history in user_histories_file], min(100, len(user_histories_file)))

# rmse_values = []
# mae_values = []
# ndcg_values = []

# ndcg_loss_fn = NDCGLoss(device)


# for user_id in tqdm(selected_users):           
#     true_ratings_list = []
#     predicted_ratings_list = []
#     predicted = []
#     predicted.extend(business_model.recommend(user_to_index[user_id]))
#     predicted.extend(user_model.recommend(user_id, user_histories_file))
#     predicted.sort(key=lambda x: x[1], reverse=True)
#     predicted = predicted[:10]
#     for item, predicted_ratings in predicted:
#         matching_rows = bussiness_data[bussiness_data['business_id'] == item]
#         if not matching_rows.empty:
#             true_ratings = matching_rows.iloc[0][2]
#         else:
#             true_ratings = None
#         true_ratings_list.append(true_ratings)
#         predicted_ratings_list.append(predicted_ratings)

#     # Calculate RMSE and MAE
#     rmse = np.sqrt(mean_squared_error(true_ratings_list, predicted_ratings_list))
#     rmse_values.append(rmse)
#     mae = mean_absolute_error(true_ratings_list, predicted_ratings_list)
#     mae_values.append(mae)
#     ndcg_loss = ndcg_loss_fn(np.array([true_ratings_list]), np.array([predicted_ratings_list]))
#     ndcg_values.append(ndcg_loss)

# # Calculate average RMSE, MAE, and NDCG loss
# average_rmse = np.mean(rmse_values)
# average_mae = np.mean(mae_values)
# average_ndcg_loss = np.mean(ndcg_values)

# print(f'Average RMSE: {average_rmse}')
# print(f'Average MAE: {average_mae}')
# print(f'Average NDCG Loss: {average_ndcg_loss}')