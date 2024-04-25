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

has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"

cwd = os.getcwd()
bussiness_index = AnnoyIndex(128, 'angular')  # 50 is the dimensionality of your item embeddings
bussiness_index.load('ann/yelp_business.ann')
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
        self.fc1 = nn.Linear(embedding_size * 2, 1024)
        x = torch.cat([user_embedding, business_embedding], dim=1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze()


    def get_score(self, business_index, distance):
        business_embeddings = self.business_embedding(business_index).to(device)
        x = torch.cat([business_embeddings, distance.view(-1, 1).to(device)], dim=1)
        self.fc1 = nn.Linear(81, 1024)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        score = torch.sigmoid(x)
        return score.squeeze()
    
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
            similar_items = bussiness_index.get_nns_by_vector(self.item_embedding(torch.tensor([business_to_index[item]])).squeeze().detach().numpy(), num_recommendations, include_distances=True)           
            for similar_item, similarity_score in zip(*similar_items):
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

class bussiness_model(nn.Module):
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

    def recommend(self, user_id, num_recommendations=10):
        user_history = self.recommendation_model.user_histories[user_id]
        recommendation_scores = {}

        for item in user_history:
            if item == 'user_id':
                continue
            # Use the transformer model to predict the rating
            rating = predict_rating(self.transformer_model,item,self.df)
            # Normalize the rating to get the attention weight
            attention_weight = F.softmax(rating, dim=-1)
            # Use the attention weight as the distance in the recommendation model
            similar_items = self.recommendation_model.recommend(item, attention_weight, num_recommendations)
            for similar_item, similarity_score in zip(*similar_items):
                if similar_item in recommendation_scores:
                    recommendation_scores[similar_item] += similarity_score
                else:
                    recommendation_scores[similar_item] = similarity_score
        recommendation_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendation_scores[:num_recommendations]

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

def predict_rating(model, business_ids, df):
    attention_weights = []
    for business_id in tqdm(business_ids, unit="business"):
        business_id = business_id.item()
        business_id = index_to_business[business_id]
        
        comments = get_reviews_for_business(business_id, df)
        if comments == []:
            attention_weights.append(0)
            continue
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
        attention_weights.append(sum(all_predictions) / len(all_predictions))
    return attention_weights



def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, df, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        # Training
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            business_ids = batch[1].to(device)            
            labels = batch[2].to(device)
            outputs = model(business_ids)
            loss = loss_fn(outputs, labels)
            loss_train_total += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        # Validation
        torch.save(model.state_dict(), 'combined_model.pth')
        model.eval()
        loss_val_total = 0
        
        for batch in val_dataloader:
            business_ids = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(business_ids)
            loss = loss_fn(outputs, labels)
            loss_val_total += loss.item()
        # Print stats
        print(f"Train loss: {loss_train_total/len(train_dataloader)}")
        print(f"Val loss: {loss_val_total/len(val_dataloader)}")
    print("Training complete!")



# Load the DataFrame from the JSON file
df = pd.read_json('yelp/processed_review.json',lines=True)

# Load the trained model
transformer_model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=5)
transformer_model.load_state_dict(torch.load('trained_model/trained_model_20240422015913.pth'))
transformer_model.eval()
transformer_model.to(device)
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')


user_to_index, business_to_index, index_to_business = fu.index_transformer()
user_histories_file = fu.json_transform(os.path.join(cwd, 'yelp/process_user.json'))
embedded_size = 128
model = RecommendationModel(user_histories_file, embedded_size, len(business_to_index))
model.load_state_dict(torch.load(os.path.join(cwd, 'trained_model/Item-based_model.pth')))
model = model.to(device)

combined_model = bussiness_model(transformer_model, model,df)
combined_model = combined_model.to(device)

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
total_len = len(dataset)
train_len = int(total_len * 0.01)
val_len = int(total_len * 0.005)
unused_len = total_len - train_len - val_len

# Create random splits
train_dataset, val_dataset, unused_dataset = random_split(dataset, [train_len, val_len, unused_len])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=48)
val_dataloader = DataLoader(val_dataset, batch_size=48)

optimizer = torch.optim.AdamW(combined_model.parameters(), lr=2e-5)
loss_fn = NDCGLoss()
train_model(combined_model, train_dataloader, val_dataloader, loss_fn, optimizer, device, df, epochs=10)
# Save the model