import os
import pickle
from annoy import AnnoyIndex
import torch
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import nn
import torch.nn.functional as F
import Frequently_Used as fu
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random



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
        self.item_ids = df['item_id'].tolist()

    def __getitem__(self, idx):
        review = self.reviews[idx]
        item_id = self.item_ids[idx]
        label = self.labels[idx]
        inputs = tokenizer.encode_plus(
            f"{item_id} {review}",
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



class ItemCollabModel(nn.Module):
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

    def recommend(self,transformer_model,df, attention_weight, user_id, num_recommendations=20):
        user_history = self.user_histories[user_id]
        recommendation_scores = {}
        for item in user_history:
            # Get the most similar items
            if item == 'user_id':
                continue
            similar_items = bussiness_index.get_nns_by_vector(self.item_embedding(torch.tensor([item_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)           
            for similar_item, similarity_score in zip(*similar_items):
                item_rating = (predict_rating(transformer_model,index_to_item[similar_item],df)[0]-1)/4
                item_rating = torch.tensor(item_rating)
                # Normalize the rating to get the attention weight
                similarity_score *= attention_weight * item_rating 
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

class ItemModel(nn.Module):
    def __init__(self, transformer_model, recommendation_model, df):
        super().__init__()
        self.transformer_model = transformer_model
        self.recommendation_model = recommendation_model
        self.df = df
    
    def forward(self, item_id):
        # Use the transformer model to predict the rating
        item_rating = predict_rating(self.transformer_model,item_id,self.df)
        item_rating = torch.tensor(item_rating)
        # Normalize the rating to get the attention weight
        attention_weight = F.softmax(item_rating, dim=-1)
        # Use the attention weight as the distance in the recommendation model
        score = self.recommendation_model.get_score(item_id, attention_weight)
        return score

    def getscore(self,item_id, attention_weight):
        score = self.recommendation_model.get_score(item_id, attention_weight)
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
            attention_weight = (rating-1) / 4  # Normalize the rating
            # Use the attention weight as the distance in the recommendation model
            similar_items = self.recommendation_model.recommend(self.transformer_model,self.df,attention_weight,user_id, num_recommendations)
            for similar_item, similarity_score in similar_items:
                similar_item = index_to_item[similar_item]
                similarity_score *= rating
                if similarity_score >5: #up
                    similarity_score = torch.tensor(5)
                similarity_score = similarity_score.item() / float(len(user_history)-1)

                # Use dict.get() to avoid KeyError
                recommendation_scores[similar_item] = recommendation_scores.get(similar_item, 0) + similarity_score 
        recommendation_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

        return recommendation_scores[:num_recommendations]
   
class UserCollabModel(nn.Module):
    def __init__(self, num_users, num_items, embedded_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedded_size)
        self.business_embedding = nn.Embedding(num_items, embedded_size)
        self.fc1 = nn.Linear(160, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.business_embedding(items)
 
        x = torch.cat([user_embedding, item_embedding], dim=1)
        self.fc_transform = nn.Linear(160, 81)
        x = self.fc_transform(x)
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
            similar_items = bussiness_index.get_nns_by_vector(self.business_embedding(torch.tensor([item_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)           
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


    def get_score(self, item_index, distance):
        item_embeddings = self.business_embedding(item_index).to(device)
        x = torch.cat([item_embeddings, distance.view(-1, 1).to(device)], dim=1)
        self.fc1 = nn.Linear(81, 1024)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()

 
class UserModel(nn.Module):
    def __init__(self, transformer_model, user_model, df):
        super().__init__()
        self.transformer_model = transformer_model
        self.user_model = user_model
        self.df = df

    def forward(self, item_id):
        # Use the transformer model to predict the rating
        rating = predict_rating(self.transformer_model,item_id,self.df)
        rating = torch.tensor(rating)
        # Normalize the rating to get the attention weight
        attention_weight = F.softmax(rating, dim=-1)
        # Use the attention weight as the distance in the recommendation model
        score = self.user_model.get_score(item_id, attention_weight)
        return score

    def getscore(self,user_id,item_id, attention_weight):
        score = self.user_model.get_score(item_id, attention_weight)
        return score
    
    def recommend(self, user_id,user_histories_file, num_recommendations=20):
        user_history = next((history for history in user_histories_file if history['user_id'] == user_id), None)
        if user_history is None:
            print(f'User {user_id} not found in the history.')
            return []

        if user_id in user_to_index and user_to_index[user_id] != -1 and user_to_index[user_id] < self.user_model.user_embedding.weight.size(0):
            user_index = torch.tensor([user_to_index[user_id]] * len(item_to_index), dtype=torch.long)
            item_indices = torch.tensor(list(item_to_index.values()), dtype=torch.long)
            rating = self.user_model(user_index, item_indices)
            rating = torch.sigmoid(rating) * 4 + 1  # Convert the ratings to a range of 1-5
            # Use the attention weight as the distance in the recommendation model
            
            top_values, top_indices = torch.topk(rating, num_recommendations)
            top_recommendations = {}
            for index, value in zip(top_indices.detach(), top_values.detach().cpu().numpy()):
                attention_weights = predict_rating(self.transformer_model, index_to_item[index.item()], self.df)
                attention_weight = np.mean(attention_weights)
                attention_weight = (attention_weight -1) / 4
                
                top_recommendations[index_to_item[index.item()]] = value * attention_weight

            top_recommendations = sorted(top_recommendations.items(), key=lambda x: x[1], reverse=True)
            return top_recommendations


def load_user_item_pairs(address):
    data = fu.json_transform(address)
    user_item_pairs = []
    for user_data in data:
        user_id = user_data['user_id']
        for item_id, rating in user_data.items():
            if item_id != 'user_id':
                user_item_pairs.append((user_id, item_id, rating))
    return user_item_pairs

def get_reviews_for_item(item_id, df):
    reviews = df[df['item_id'] == item_id]['text'].tolist()
    return reviews

def predict_rating(model, item_id, df):
    attention_weights = []
    comments = get_reviews_for_item(item_id, df)
    df1 = pd.DataFrame({
        'text': comments,
        'item_id': [item_id] * len(comments),
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


    
# get rating of the user past item record
def get_item_score(user_id, item_id, user_model,item_model, transformer_model, df):
    # Use the transformer model to predict the rating
    rating = predict_rating(transformer_model, item_id, df)
    rating = torch.tensor(rating)
    # Convert item_id to item_index
    item_index = item_to_index[item_id]
    # Use the transformer model to predict the rating
    predicted_rating = predict_rating(transformer_model, item_id, df)
    predicted_rating = torch.tensor(predicted_rating)
    # Normalize the predicted rating to get the attention weight
    attention_weight = F.softmax(predicted_rating, dim=-1)
    item_index = torch.tensor([item_index]).to(device)
    # Use the attention weight as the distance in the recommendation model
    score = (user_model.getscore(user_id, item_index, attention_weight) + item_model.getscore(item_index, attention_weight)) / 2
    score = score.item()
    return score

# cal the loss of the user by past item record, if number_of_recommender == 1, and user_id is filled, it will generate the selected user
def Loss_Cal(number_of_recommender, user_id=None):
    if user_id is not None and number_of_recommender == 1:
        selected_users = [user_id]
    else:
        selected_users = random.sample([user_history['user_id'] for user_history in user_histories_file], min(number_of_recommender, len(user_histories_file)))

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
        # For each item in the user history, predict the score and compare it with the actual rating
        for item_id, true_rating in user_history.items():
            if item_id == 'user_id':
                continue
            predicted_rating = get_item_score(user_id, item_id, user_model, item_model, transformer_model, df)
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

    return average_rmse, average_mae, average_ndcg_loss


# input the user_id and number of recommender need, return the list of the predicted ratings
def get_ratings(selected_user_id,number_of_recommender):
    predicted = []
    predicted.extend(item_model.recommend(user_to_index[selected_user_id],number_of_recommender))
    predicted.extend(user_model.recommend(selected_user_id, user_histories_file,number_of_recommender))
    predicted.sort(key=lambda x: x[1], reverse=True)
    predicted = predicted[:number_of_recommender]
    return predicted





has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "cpu"

cwd = os.getcwd()
bussiness_index = AnnoyIndex(128, 'angular') 
bussiness_index.load('ann/yelp_item.ann')
# Load the mappings
with open('pkl/user_to_index.pkl', 'rb') as f: 
    user_to_index = pickle.load(f)
with open('pkl/item_to_index.pkl', 'rb') as f:
    item_to_index = pickle.load(f)
with open('pkl/index_to_item.pkl', 'rb') as f:
    index_to_item = pickle.load(f)
# Load the DataFrame from the JSON file
df = pd.read_json('data/Processed_Review.json',lines=True)
bussiness_data = pd.read_json('data/Processed_Item_Data.json',lines=True)

# Load the trained model
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
user_to_index, item_to_index, index_to_item = fu.index_transformer()
user_histories_file = fu.json_transform(os.path.join(cwd, 'data/Process_Interact_History.json'))
item_embedded_size = 128
user_embedded_size = 80
transformer_model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=5)
item_based_model = ItemCollabModel(user_histories_file, item_embedded_size, len(item_to_index))
user_based_model = UserCollabModel(len(user_to_index), len(item_to_index), user_embedded_size)

transformer_model.load_state_dict(torch.load('trained_models/Transformer.pth'))
item_based_model.load_state_dict(torch.load(os.path.join(cwd, 'trained_models/Item_model.pth')))
user_based_model.load_state_dict(torch.load(os.path.join(cwd, 'trained_models/User_model.pth')))


item_based_model = item_based_model.to(device)
user_based_model = user_based_model.to(device)
item_model = ItemModel(transformer_model, item_based_model,df)
user_model = UserModel(transformer_model, user_based_model,df)
transformer_model.eval().to(device)
item_model = item_model.eval().to(device)
user_model = user_model.eval().to(device)

# Load mappings
user_item_pairs = load_user_item_pairs(os.path.join(cwd, 'data/Process_Interact_History.json'))
# Convert the list to a DataFrame
user_item_pairs_df = pd.DataFrame(user_item_pairs, columns=['user_id', 'item_id', 'rating'])
# Now you can call to_records on the DataFrame
user_item_pairs_list = [tuple(x) for x in user_item_pairs_df.to_records(index=False)]
user_to_index = {user: i for i, user in enumerate(set(user_id for user_id, _, _ in user_item_pairs_list))}
item_to_index = {item: i for i, item in enumerate(set(item_id for _, item_id, _ in user_item_pairs_list))}


#testcase for the function
average_rmse, average_mae, average_ndcg_loss = Loss_Cal(1)
print(f'Average RMSE: {average_rmse}')
print(f'Average MAE: {average_mae}')
print(f'Average NDCG Loss: {average_ndcg_loss}')

user_id ='--034gGozmK4y5txuPsdAA'
average_rmse, average_mae, average_ndcg_loss = Loss_Cal(1,user_id)
print(f'Average RMSE: {average_rmse}')
print(f'Average MAE: {average_mae}')
print(f'Average NDCG Loss: {average_ndcg_loss}')

predicted_ratings = get_ratings(user_id, 10)
for rating in predicted_ratings:
    print(f'Item ID: {rating[0]}, Predicted Rating: {rating[1]}')