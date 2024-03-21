import time
from models import cnn_colab
from CNN_support_lib import default_item_based_dataset as item_based_dataset

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import json
import os
from torchinfo import summary

def main():
    cwd = os.getcwd()  # Set current working directory
    # Load JSON data from a file
    data_path = os.path.join(cwd, 'data')
    l3_path = os.path.join(data_path, 'l3_data')
    train_cat = 'Industrial_and_Scientific'
    asin_list = []
    preset_data = []  # add read size
    with open(os.path.join(l3_path, 'processed_{}.json'.format(train_cat)), 'r') as f:
        for line in f: 
            preset_data.append(json.loads(line))

    with open(os.path.join(l3_path, 'asin_label_{}.yaml'.format(train_cat)), 'r') as f:
        for line in f:
            asin_list.append(line.strip())

    dataset = item_based_dataset(preset_data, asin_list)
    print("Maxlen: ", dataset.__maxlen__())
    print("Asinlen: ", dataset.__asinlen__())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # show one data
    # for i, (data, label) in enumerate(dataloader):
    #     print(data)
    #     print(label)
    #     break

    # Set the device to use
    if not torch.cuda.is_available():
        print("CUDA driver is not installed.")
    else:
        print("CUDA driver is installed.")
    if torch.backends.cudnn.is_available():
        print("cuDNN is installed.")
    else:
        print("cuDNN is not installed.")
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Using device: {device}")
    # Initialize the model
    model = cnn_colab.CNN_Collaborative_Filtering(len(asin_list))
    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    # Initialize the loss function
    criterion = nn.MSELoss()
    
    # # Print a test model summary
    # test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # try:
    #     summary(model=model, input_size=test_input.shape, col_width=15,
    #     col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    #     row_settings=['var_names'], verbose=0)
    # except Exception as e:
    #     # print(f"An error occurred: {e}")
    #     raise e

    # Train the model
    print("Model: ", model.__class__.__name__)
    model.to(device)
    torch.cuda.empty_cache()
    
    for epoch in range(10):
        time_ckpt = time.time()
        running_loss = 0.0
        model.train()
        __t = time.time()
        for i, (data, label) in enumerate(dataloader):
            label = label.to(device)
            data = data[-1].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # print(f"Epoch {epoch + 1} batch {i + 1} loss: {loss.item()}")
            if time.time() - __t > 5 or i == 0 or i == len(dataloader) - 1:
                print(f"Epoch {epoch + 1} batch {i + 1} loss: {loss.item()}")
                __t = time.time()
        
        print(f"Epoch {epoch + 1} loss: {running_loss / len(dataloader)} time: {time.time() - time_ckpt}")

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    
if __name__ == "__main__":
    main()