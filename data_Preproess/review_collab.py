import json
from tqdm import tqdm

# Create a dictionary to map word numbers to integers
word_to_number = {
    'OneStar': 1,
    'TwoStar': 2,
    'ThreeStar': 3,
    'FourStar': 4,
    'FiveStar': 5
}

counter = 0

with open('yelp/processed_review3.json') as f, open('process_user.json', 'w') as out_f:
    for line in tqdm(f, desc="Processing users"):
        # Load each JSON object separately
        user = json.loads(line)

        user_id = user['user_id']
        # Create a new dictionary for each user
        new_user = {
            'user_id': user_id
        }
        for star, businesses in user['reviews'].items():
            for business_id in businesses:
                # Convert the word number to an integer
                star_number = word_to_number[star]

                new_user[business_id] = (star_number) 

        # Write the new user to the JSON file
        json.dump(new_user, out_f)
        out_f.write('\n')

        # Increment the counter
        counter += 1
                
        # Break if the counter reaches 100