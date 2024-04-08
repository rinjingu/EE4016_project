import json
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import os

timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
folder_name = f"results_{timestamp}"
os.makedirs(folder_name)

# Load the JSON file
with open('processed_review2.json', 'r') as f:
    data = []
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Error decoding line: {line}")
            continue

# Convert the list of JSON objects to a DataFrame
df = pd.DataFrame(data)

# Load the RoBERTa model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print("Model and tokenizer prepared")

# Create a custom dataset class
class ReviewDataset(Dataset):
    def __init__(self, df):
        self.reviews = df['text'].tolist()
        self.labels = df['stars'].tolist()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        inputs = tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label - 1, dtype=torch.long)  # Adjust labels to be in the range [0, 4]
        }

# Create the dataset and dataloader
dataset = ReviewDataset(df)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Set the device to CUDA if available
device = torch.device('cuda:0')

# Move the model and optimizer to the device
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

print("Start training")
# Train the model
total_loss = 0
for epoch in range(5):
    print(f"Epoch {epoch+1}/5")
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

# Calculate the average mean loss
avg_loss = total_loss / len(dataloader)
print(f'Average mean loss: {total_loss} / {len(dataloader)} = {avg_loss:.4f}')

# Save the trained model with the timestamp
model_filename = os.path.join(folder_name, f"trained_model_{timestamp}.pth")
torch.save(model.state_dict(), model_filename)

print("Start evaluation")
# Evaluate the model on the entire dataset
model.eval()
all_predictions = []
with tqdm(dataloader, unit="batch") as teval:
    for batch in teval:
        teval.set_description("Evaluating")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1) + 1  # Adjust predictions back to the range [1, 5]
        all_predictions.extend(predictions.tolist())

# Add the predictions to the original DataFrame
df['roberta_score'] = all_predictions
df_filename = os.path.join(folder_name, f"review_scores_full_{timestamp}.json")
df.to_json(df_filename, orient='records', lines=True)

# Create a new DataFrame with selected columns
new_df = df[['review_id', 'stars', 'roberta_score']]

# Save the new DataFrame to a JSON file
new_df_filename = os.path.join(folder_name, f"review_scores_{timestamp}.json")
new_df.to_json(new_df_filename, orient='records', lines=True)
print("New JSON files created")