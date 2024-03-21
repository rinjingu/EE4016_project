import torch
import torch.nn as nn
from torch.utils.data import Dataset

import json
import os


class default_item_based_dataset(Dataset):
    def __init__(self, preset_data, asin_list):
        self.preset_data = preset_data
        self.asin_list = asin_list
        self.max_len = self.__maxlen__()

    def __getitem__(self, index):
        try:
            asin = self.preset_data[index]['asin']
            price_from = float(self.preset_data[index]['price'][0]) or -1
            price_to = float(self.preset_data[index]['price'][1]) or -1
            avg_rating = self.preset_data[index]['avg_rating'] or -1
            rank = self.preset_data[index]['rank'] or -1
            brand = self.preset_data[index]['brand'] or -1
            category = self.preset_data[index]['category'] or [-1, -1]
            category_a, category_b = (category + [-1, -1])[:2]
            activeness = self.preset_data[index]['activeness'] or -1
            relation = self.preset_data[index]['relation'] or {}
            fill_relation = []

            for i in list(relation.keys()):
                fill_relation.append(int(i))

            # if len(fill_relation) < 1:
            #     fill_relation.append(len(self.asin_list))
            while len(fill_relation) < self.max_len:
                fill_relation.append(len(self.asin_list))

            packed_data = [price_from, price_to, rank, brand,
                           category_a, category_b, activeness, fill_relation]
            trainset = [torch.tensor(data).long() for data in packed_data]   # convert to tensor
            Verify_ans = torch.tensor([avg_rating], dtype=torch.float)
            return trainset, Verify_ans
        except KeyError as e:
            print(f"KeyError: {e} is not found in the data at index {index}")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e

    def __len__(self):
        return len(self.preset_data)
    
    def __maxlen__(self):
        return max([len(data["relation"]) for data in self.preset_data])
    
    def __asinlen__(self):
        return len(self.asin_list)

class item_based_dataset_with_relation_v1(Dataset):
    def __init__(self, preset_data):
        self.preset_data = preset_data

    def __getitem__(self, index):
        try:
            price_from = float(self.preset_data[index]['price'][0]) or -1
            price_to = float(self.preset_data[index]['price'][1]) or -1
            avg_rating = self.preset_data[index]['avg_rating'] or -1
            rank = self.preset_data[index]['rank'] or -1
            brand = self.preset_data[index]['brand'] or -1
            category = self.preset_data[index]['category'] or [-1, -1]
            category_a, category_b = (category + [-1, -1])[:2]
            activeness = self.preset_data[index]['activeness'] or -1
            relation = self.preset_data[index]['relation'] or {}
            packed_data = [price_from, price_to, rank, brand,
                           category_a, category_b, activeness, relation]
            trainset = [torch.tensor(data) for data in packed_data]  # convert to tensor
            Verify_ans = torch.tensor([avg_rating], dtype=torch.float)
            return trainset, Verify_ans
        except KeyError as e:
            print(f"KeyError: {e} is not found in the data at index {index}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __len__(self):
        return len(self.preset_data)

if __name__ == "__main__":
    print("This is a support library, not the main program")
