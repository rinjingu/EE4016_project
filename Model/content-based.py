import json
from annoy import AnnoyIndex
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam

class NDCGLoss(nn.Module):
    def _init_(self):
        super()._init_()

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

class ContentModel(nn.Module):
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

def get_recommendations(model, item_id, id_to_index, index_to_id, top_k=10):
    item_index = torch.tensor([id_to_index[item_id]], dtype=torch.long)
    vector = model(item_index)
    scores = (model(torch.arange(len(id_to_index))) * vector).sum(dim=1)
    top_indices = scores.topk(top_k+1).indices
    return [index_to_id[i.item()] for i in top_indices if i != item_index]

if __name__ == "__main__":
    # Load the data
    with open('yelp/for_item.json', 'r') as f:
        data = [json.loads(line) for line in tqdm(f, desc="Loading data")]
    df = pd.DataFrame(data)

    # Create mappings from item_id to index and vice versa
    id_to_index = {id: index for index, id in enumerate(df['item_id'])}
    index_to_id = {index: id for id, index in id_to_index.items()}

    # Define the model
    embedding_size = 128
    model = ContentModel(len(id_to_index), embedding_size)

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
        
        # Extract item embeddings from the trained model
    item_indices = torch.arange(len(id_to_index), dtype=torch.long)
    embeddings = model.item_embedding(item_indices).detach().numpy()

    # Initialize an Annoy index
    t = AnnoyIndex(embeddings.shape[1], 'angular')  # Length of item vector that will be indexed

    # Add each item embedding to the Annoy index
    for i, embedding in enumerate(embeddings):
        t.add_item(i, embedding)

    # Build the Annoy index
    t.build(128)  # 128 trees

    # Save the Annoy index to a file
    t.save('yelp_item.ann')
    # Query the index for the 10 nearest neighbors of the first item
    nearest_neighbors = t.get_nns_by_item(0, 10)

    # Print the nearest neighbors
    print("Nearest neighbors of the first item:", nearest_neighbors)


    # Get recommendations for a user-specified item_id
    while True:
        item_id = input("Enter a item_id: ")
        if item_id in id_to_index:
            recommendations = get_recommendations(model, item_id, id_to_index, index_to_id)
            print("Recommended items:", recommendations)
        else:
            print("item_id not found. Please try again.")
