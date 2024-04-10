import json
import pickle

# # Load the JSON data
# user_ids = []
# with open('yelp/processed_user.json', 'r') as f:
#     for line in f:
#         item = json.loads(line)
#         user_ids.append(item['user_id'])

# # Create a dictionary mapping business IDs to unique integer indices
# user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

# while len(user_to_index) < 1987929:
#     user_to_index[len(user_to_index)] = -1

# # Save the dictionary to a pickle file
# with open('pkl/user_to_index.pkl', 'wb') as f:
#     pickle.dump(user_to_index, f)
    
# Load the JSON data
business_ids = []
with open('yelp/processed_business.json', 'r') as f:
    for line in f:
        item = json.loads(line)
        business_ids.append(item['business_id'])



# Create a dictionary mapping business IDs to unique integer indices
business_to_index = {business_id: index for index, business_id in enumerate(business_ids)}
while len(business_to_index) < 150346:
    business_to_index[len(business_to_index)] = -1
index_to_business = {index: business_id for business_id, index in business_to_index.items()}

# Save the dictionary to a pickle file
with open('pkl/business_to_index.pkl', 'wb') as f:
    pickle.dump(business_to_index, f)
    
with open('pkl/index_to_business.pkl', 'wb') as f:
    pickle.dump(index_to_business, f)