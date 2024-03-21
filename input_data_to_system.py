# import package
import torch
import torch.nn as nn
import json,os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

cwd = os.getcwd() # Set current working directory
# Load JSON data from a file
preset_data = [] #add read size
with open(os.path.join(cwd, 'data.json'), 'r') as f:
    for line in f:
        preset_data.append(json.loads(line))

class item_based_dataset(Dataset):
    def __init__(self, preset_data):
        self.preset_data = preset_data

    def __getitem__(self, index):
        try:
            asin = self.preset_data[index]['asin']
            price_mean = float(self.preset_data[index]['price'][0]) or -1
            if float(self.preset_data[index]['price']) == 0:
                price_diff = 0
            else:
                price_diff = float(self.preset_data[index]['price'][1]) or -1
            avg_rating = self.preset_data[index]['avg_rating'] or -1
            rank = self.preset_data[index]['rank'] or -1
            brand = self.preset_data[index]['brand'] or -1
            category = self.preset_data[index]['category'] or [-1, -1]
            category_a, category_b = (category + [-1, -1])[:2]
            activeness = self.preset_data[index]['activeness'] or -1
            packed_data = [price_mean,price_diff,rank,brand,category_a,category_b,activeness]
            trainset = torch.tensor(packed_data)  # convert to tensor
            Verify_ans = torch.tensor([avg_rating], dtype=torch.float)
            return trainset,Verify_ans
        except KeyError as e:
            print(f"KeyError: {e} is not found in the data at index {index}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __len__(self):
        return len(self.preset_data)
    

dataset = item_based_dataset(preset_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
