import pickle
from annoy import AnnoyIndex
import torch
import os
import Frequently_Used as fu
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()
bussiness_index = AnnoyIndex(128, 'angular')  # 50 is the dimensionality of your item embeddings
bussiness_index.load('yelp_item.ann')
with open('pkl/user_to_index.pkl', 'rb') as f:
    user_to_index = pickle.load(f)
with open('pkl/item_to_index.pkl', 'rb') as f:
    item_to_index = pickle.load(f)
with open('pkl/index_to_item.pkl', 'rb') as f:
    index_to_item = pickle.load(f)
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
        score = torch.sigmoid(self.linear4(x))
        return score.squeeze()

    def recommend(self, user_id, num_recommendations=10):
        user_history = self.user_histories[user_id]
        recommendation_scores = {}

        for item in user_history:
            # Get the most similar items
            if item == 'user_id':
                continue
            similar_items = bussiness_index.get_nns_by_vector(self.item_embedding(torch.tensor([item_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)           
            for similar_item, similarity_score in zip(*similar_items):
                if similar_item in recommendation_scores:
                    recommendation_scores[similar_item] += similarity_score
                else:
                    recommendation_scores[similar_item] = similarity_score
        recommendation_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendation_scores[:num_recommendations]

user_to_index, item_to_index, index_to_item = fu.index_transformer()
user_histories_file = fu.json_transform(os.path.join(cwd, 'yelp/process_user.json'))
embedded_size = 128
model = RecommendationModel(user_histories_file, embedded_size, len(item_to_index))
model.load_state_dict(torch.load(os.path.join(cwd, 'model_epoch_1.pth')))
model = model.to(device)

user_id = '0VqczoKSmj65LHqfo0jRaQ'

recommendations = model.recommend(user_to_index[user_id])
for item_id, score in recommendations:
    item_id = index_to_item[item_id]
    print(f'item ID: {item_id}, Score: {score}')