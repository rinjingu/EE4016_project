import json
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam

class MLPModel(nn.Module):
    def __init__(self, num_items, embedding_size):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size, 5192)
        self.fc2 = nn.Linear(5192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, embedding_size)

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

def get_recommendations(model, business_id, id_to_index, index_to_id, top_k=10):
    item_index = torch.tensor([id_to_index[business_id]], dtype=torch.long)
    vector = model(item_index)
    scores = (model(torch.arange(len(id_to_index))) * vector).sum(dim=1)
    top_indices = scores.topk(top_k+1).indices
    return [index_to_id[i.item()] for i in top_indices if i != item_index]

if __name__ == "__main__":
    # Load the data
    with open('processed3.json', 'r') as f:
        data = [json.loads(line) for line in tqdm(f, desc="Loading data")]
    df = pd.DataFrame(data)

    # Create mappings from business_id to index and vice versa
    id_to_index = {id: index for index, id in enumerate(df['business_id'])}
    index_to_id = {index: id for id, index in id_to_index.items()}

    # Define the model
    embedding_size = 50
    model = MLPModel(len(id_to_index), embedding_size)

    # Define the loss function and the optimizer with L2 regularization
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), weight_decay=0.01)  # weight_decay parameter adds L2 regularization

    # Initialize variables for Early Stopping
    min_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Train the model
    for epoch in tqdm(range(100), desc="Training"):
        item_indices = torch.arange(len(id_to_index), dtype=torch.long)
        vectors = model(item_indices)
        loss = criterion(vectors, torch.ones_like(vectors))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get recommendations for a user-specified business_id
    while True:
        business_id = input("Enter a business_id: ")
        if business_id in id_to_index:
            recommendations = get_recommendations(model, business_id, id_to_index, index_to_id)
            print("Recommended businesses:", recommendations)
        else:
            print("Business_id not found. Please try again.")
