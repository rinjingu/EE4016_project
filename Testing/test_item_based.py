import torch
import os
import pickle
import EE4016_project.Model.bussiness_based_collab as bbc
import EE4016_project.Frequently_Used as fu
from torch.utils.data import DataLoader
    # Save user-item pairs




def test_recommendation_model():
    cwd = os.getcwd()
    embeeded_size = 128
    business_to_index = fu.index_transformer()[1]  # assuming this function returns business_to_index as the second return value

    # Load user-item pairs
    with open('user_item_pairs.pkl', 'rb') as f:
        user_item_pairs = pickle.load(f)

    # Load business-to-index mapping
    with open('business_to_index.pkl', 'rb') as f:
        business_to_index = pickle.load(f)
    # Load your pretrained model
    model = bbc.RecommendationModel(os.path.join(cwd, 'yelp/process_user.json'), embeeded_size, len(business_to_index))
    model.load_state_dict(torch.load('model_epoch_5.pth'))  # replace with your model file
    model.eval()

    # Load data
    user_item_pairs = fu.load_user_item_pairs(os.path.join(cwd, 'yelp/process_user.json'))  # assuming this function loads user-item pairs
    dataset = bbc.UserItemDataset(user_item_pairs, business_to_index)
    data_loader = DataLoader(dataset, batch_size=128)

    # Specify the user_id
    user_id = 'example_user_id'  # replace with your user_id

    # Generate recommendations
    recommendations = model.recommend(user_id, num_recommendations=10)

    # Print the recommendations
    print(recommendations)

if __name__ == "__main__":
    test_recommendation_model()