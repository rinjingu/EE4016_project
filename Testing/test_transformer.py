import torch
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the DataFrame from the JSON file
df = pd.read_json('yelp/processed_review.json',lines=True)
# Load the trained model
model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=5)
model.load_state_dict(torch.load('trained_model/trained_model_20240422015913.pth'))
model.eval()

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

# Define the custom dataset class
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

# Set the device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model and optimizer to the device
model.to(device)

def get_reviews_for_item(item_id, df):
    return df[df['item_id'] == item_id]['text'].tolist()

def predict_rating(model,item_id, df):
    comments = get_reviews_for_item(item_id, df)
    df = pd.DataFrame({
        'text': comments,
        'item_id': [item_id] * len(comments),
        'stars': [0] * len(comments)  # Dummy labels
    })
    dataset = ReviewDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    all_predictions = []
    with tqdm(dataloader, unit="batch") as teval:
        for batch in teval:
            teval.set_description("Evaluating")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1) + 1
            all_predictions.extend(predictions.tolist())
    return sum(all_predictions) / len(all_predictions)

# Prompt the user for a item ID and predict the rating
while True:
    item_id = input("Enter a item ID or 'q' to quit: ")
    if item_id.lower() == 'q':
        break
    predicted_rating = predict_rating(model,item_id, df)
    print(f"The predicted rating for item {item_id} is {predicted_rating}")