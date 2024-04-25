import json
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"

class UserCollabModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 1024)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
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
    
    
if __name__ == "__main__":
    # Load the data
    with open('yelp/process_user.json', 'r') as f:
        data = [json.loads(line) for line in tqdm(f, desc="Loading data")]

    # Transform the data into a list of user-item-rating triplets
    triplets = [(user['user_id'], item, rating) for user in data for item, rating in user.items() if item != 'user_id']

    # Convert the triplets into a DataFrame
    df = pd.DataFrame(triplets, columns=['user_id', 'item_id', 'rating'])

    # Map user_ids and item_ids to integers
    user_to_index = {id: index for index, id in enumerate(df['user_id'].unique())}
    item_to_index = {id: index for index, id in enumerate(df['item_id'].unique())}
    index_to_user = {index: id for id, index in user_to_index.items()}
    index_to_item = {index: id for id, index in item_to_index.items()}

    users = torch.tensor([user_to_index[id] for id in df['user_id']], dtype=torch.long)
    items = torch.tensor([item_to_index[id] for id in df['item_id']], dtype=torch.long)
    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    plt.figure(figsize=(10, 4))
    plt.hist(df['rating'], bins=30, edgecolor='black')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()
    users_train, users_test, items_train, items_test, ratings_train, ratings_test = train_test_split(users.cpu().numpy(), items.cpu().numpy(), ratings.cpu().numpy(), test_size=0.2, random_state=42)

    users_train = torch.tensor(users_train, dtype=torch.long).to(device)
    items_train = torch.tensor(items_train, dtype=torch.long).to(device)
    ratings_train = torch.tensor(ratings_train, dtype=torch.float32).to(device)

    users_test = torch.tensor(users_test, dtype=torch.long).to(device)
    items_test = torch.tensor(items_test, dtype=torch.long).to(device)
    ratings_test = torch.tensor(ratings_test, dtype=torch.float32).to(device)
    
    # Define the model
    embedding_size = 50
    model = UserCollabModel(len(user_to_index), len(item_to_index), embedding_size)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    learning_rate = 0.001
    weight_decay = 0.01  # L2 regularization
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # Define the batch size
    batch_size = 512
    model = model.to(device)
    users_train = users_train.to(device)
    items_train = items_train.to(device)
    ratings_train = ratings_train.to(device)
    # Train the model
    # Train the model with early stopping
    best_mae = float('inf')
    no_improve_epochs = 0
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(20), desc="Training"):  # Increase the number of epochs
        permutation = torch.randperm(users_train.size()[0])
        for i in range(0, users_train.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_users, batch_items, batch_ratings = users_train[indices], items_train[indices], ratings_train[indices]
            outputs = model(batch_users, batch_items)
            loss = criterion(outputs, batch_ratings)
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        model.eval()
        with torch.no_grad():
            test_outputs = model(users_test, items_test)
            mae = mean_absolute_error(ratings_test.cpu().numpy(), test_outputs.cpu().numpy())
            rmse = mean_squared_error(ratings_test.cpu().numpy(), test_outputs.cpu().numpy(), squared=False)
        val_losses.append(mae)
        print(f'Test MAE: {mae}, Test RMSE: {rmse}')
        # Early stopping
        if mae < best_mae:
            best_mae = mae
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= 5:  # Stop training after 5 epochs without improvement
                print('Early stopping')
                break
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# # User interaction loop
# while True:
#     model = UserCollabModel(len(user_to_index), len(item_to_index), embedding_size)
#     model.load_state_dict(torch.load('model.pth'))
#     model = model.to(device)
#     mode = input("Enter 'user' to recommend items for a user, 'item' to predict the rating for a item, or 'quit' to exit: ")
#     if mode.lower() == 'quit':
#         break
#     elif mode.lower() == 'user':
#         user_id = input("Enter user id: ")
#         if user_id in user_to_index:
#             user_index = torch.tensor([user_to_index[user_id]] * len(item_to_index), dtype=torch.long).to(device)
#             item_indices = torch.tensor(list(item_to_index.values()), dtype=torch.long).to(device)
#             ratings = model(user_index, item_indices)
#             ratings = torch.sigmoid(ratings) * 4 + 1  # Convert the ratings to a range of 1-5
#             top10_values, top10_indices = torch.topk(ratings, 10)
#             top10_items = [index_to_item[index.item()] for index in top10_indices]
#             top10_ratings = top10_values.cpu().numpy()
#             print(f'Top 10 recommended items for user {user_id} and their predicted ratings:')
#             for item, rating in zip(top10_items, top10_ratings):
#                 print(f'item: {item}, Predicted rating: {rating:.2f}')
#         else:
#             print("User id not found. Please try again.")

#     elif mode.lower() == 'item':
#         item_id = input("Enter item id: ")
#         if item_id in item_to_index:
#             item_index = torch.tensor([item_to_index[item_id]], dtype=torch.long).to(device)
#             user_indices = torch.tensor(list(user_to_index.values()), dtype=torch.long).to(device)
#             # Repeat the item_index tensor to match the size of user_indices
#             item_index = item_index.repeat(user_indices.size(0))
#             ratings = model(user_indices, item_index)
#             ratings = torch.sigmoid(ratings) * 4 + 1  # Convert the ratings to a range of 1-5
#             print(f'Predicted rating for item {item_id}: {ratings.mean().item():.2f}')
#         else:
#             print("item id not found. Please try again.")
#     else:
#         print("Invalid mode. Please enter 'user' or 'item'.")