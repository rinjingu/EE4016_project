import json
import pickle

# Load the JSON data
user_ids = []
with open('yelp/processed_user.json', 'r') as f:
    for line in f:
        item = json.loads(line)
        user_ids.append(item['user_id'])

# Create a dictionary mapping item IDs to unique integer indices
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

while len(user_to_index) < 1987929:
    user_to_index[len(user_to_index)] = -1
index_to_user = {index: item_id for item_id, index in user_to_index.items()}

# Save the dictionary to a pickle file
with open('pkl/user_to_index.pkl', 'wb') as f:
    pickle.dump(user_to_index, f)
with open('pkl/index_to_user.pkl', 'wb') as f:
    pickle.dump(index_to_user, f)
    
# Load the JSON data
item_ids = []
with open('yelp/processed_item.json', 'r') as f:
    for line in f:
        item = json.loads(line)
        item_ids.append(item['item_id'])




# Create a dictionary mapping item IDs to unique integer indices
item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
while len(item_to_index) < 150346:
    item_to_index[len(item_to_index)] = -1
index_to_item = {index: item_id for item_id, index in item_to_index.items()}

# Save the dictionary to a pickle file
with open('pkl/item_to_index.pkl', 'wb') as f:
    pickle.dump(item_to_index, f)
    
with open('pkl/index_to_item.pkl', 'wb') as f:
    pickle.dump(index_to_item, f)