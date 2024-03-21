import torch
import torch.nn as nn

class CNN_Collaborative_Filtering(nn.Module):
    def __init__(self, num_items, embedding_dim : float = 16, dropout : float = 0.2):
        super(CNN_Collaborative_Filtering, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.X = nn.Embedding(num_items+1, embedding_dim)
        # self.Y = nn.Embedding(num_items, embedding_dim)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.X(inputs)
        # y = self.Y(inputs)
        x = x.unsqueeze(1)
        # y = y.unsqueeze(1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = torch.flatten(x, 1)
        fc = nn.Linear(x.shape[1], 1)
        fc = fc.to(x.device)
        x = fc(x)
        return x
    
class test_embad(nn.Module):
    def __init__(self, num_items, embedding_dim : float = 25, dropout : float = 0.2):
        super(test_embad, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.X = nn.Embedding(num_items, embedding_dim)
        # self.Y = nn.Embedding(num_items, embedding_dim)

    def forward(self, inputs):
        print("im here")
        print(inputs)
        print("im still here")
        x = self.X(inputs)
        return x