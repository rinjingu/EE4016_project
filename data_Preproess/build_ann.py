import json
import numpy as np
from annoy import AnnoyIndex

# Load the JSON file
with open('yelp/processed_business.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert the JSON objects to a format usable by Annoy
features = [np.array(list(item.values())) for item in data]

# Initialize an Annoy index
t = AnnoyIndex(len(features[0]), 'angular')  # Length of item vector that will be indexed

# Add each item to the Annoy index
for i, feature in enumerate(features):
    t.add_item(i, feature)

# Build the Annoy index
t.build(50)  # 50 trees

# Save the Annoy index to a file
t.save('yelp_business.ann')