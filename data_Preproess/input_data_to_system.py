# input_data_to_system.py
# ---------------------
#
# This module provides a simple way to load and preprocess the data from dataset.
# It is designed to be used with the `RNN` model defined in `Rnn.py`.
#
# Features
# -------
#
# * Loads data from JSON files using the `json_to_system` class.
# * Preprocesses the data by converting categorical variables into numerical variables and normalizing the price and rating values.
# * Provides a way to load category and brand labels from YAML files using the `yaml_to_system` class.
#
# Usage
# ----
#
# To use this module in your own code, simply import it and use the `json_to_system` or `yaml_to_system` classes to load the data and category labels. For example:
# ```
# import input_data_to_system as idts
#
# # Load data from JSON file
# idts.json_to_system('path/to/file.json')
#
# # Preprocess data and convert categorical variables into numerical variables
# preprocessed_data = idts.preprocess(loaded_data)
#
# # Load category labels from YAML file
# idts.yaml_to_system('path/to/labels.yml')
# 
# # Create dataset and dataloader
#dataset = input_data.item_based_dataset(preset_dataset.__set_data__())
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for epoch in range(10):
#    for x, y in dataloader:
        # Your model code here...
#        optimizer.zero_grad()
#        outputs = model(x)
#        loss = criterion(outputs, y)
#        loss.backward()
#        optimizer.step()
#    print('Epoch {}: Loss {:.4f}'.format(epoch+1, loss.item()))

# In this example, we first create an instance of the 
# `input_data.json_to_system` class and load JSON data from a file using the
# `__set_data__()` method. We then create an instance of the 
# `input_data.yaml_to_system` class and load YAML data from a file using the
# `__set_data__()` method.

# Next, we create an instance of the `input_data.item_based_dataset` class 
# and pass it the JSON data that was loaded by the `json_to_system` class. 
# We then create a PyTorch dataloader from this dataset using the 
# `DataLoader` function.

# Finally, we iterate over the dataloader in an epoch-wise manner, applying 
# the model to each batch of input data and calculating the loss for each 
# iteration. We print the current epoch number and the loss for each 
# iteration.




# import package
import torch
import json,os,yaml
from torch.utils.data import Dataset, DataLoader

cwd = os.getcwd() # Set current working directory
has_mps = torch.backends.mps.is_built()
device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"

class json_to_system():
    def __init__(self,filename):
        self.preset_data = [] #add read size
        self.filename = filename
    # Load JSON data from a file
    def __set_data__(self):
        with open(os.path.join(cwd, self.filename), 'r') as f:
            for line in f:
                self.preset_data.append(json.loads(line))          
        return self.preset_data

    
    def __get_asin__(self,collumn):
        return self.preset_data[collumn]['asin'].__getattribute__
    
    def __get_all__(self,collumn):
        return self.preset_data[collumn]
    
class yaml_to_system():
    def __init__(self,filenames):
        self.meanings = []
        with open(os.path.join(cwd,filenames), 'r') as f:
            for line in f:
                self.meanings.append(yaml.safe_load(line))          
        return self.meanings         

def json_transform(address):    
    data = []
    with open(address, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
    
class item_based_dataset(Dataset):
    def __init__(self, preset_data):
        self.preset_data = preset_data
        self.packed_data = []

    def __getitem__(self, index):

        # Save the preset data to tensor
        asin = self.preset_data[index]['asin']
        price_mean = float(self.preset_data[index]['price'][0]) or -1
        if float(self.preset_data[index]['price'][1]) == 0:
            price_diff = 0
        else:
            price_diff = float(self.preset_data[index]['price'][1]) or -1
            
        avg_rating = self.preset_data[index]['avg_rating'] or -1
        rank = self.preset_data[index]['rank'] or -1
        brand = self.preset_data[index]['brand'] or -1
        category = self.preset_data[index]['category'] or [-1, -1]
        category_a, category_b = (category + [-1, -1])[:2]
        activeness = self.preset_data[index]['activeness'] or -1
        self.packed_data = [price_mean,price_diff,rank,brand,category_a,category_b,activeness]

        trainset = torch.tensor(self.packed_data)  # convert to tensor
        Verify_ans = torch.tensor([avg_rating], dtype=torch.float) #
        
        return trainset,Verify_ans
        
    

    def __len__(self):
        return len(self.preset_data)
    


