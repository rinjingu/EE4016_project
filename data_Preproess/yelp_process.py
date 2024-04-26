import json


    
# # Path to the input JSON file
# # input_file = 'data/yelp/yelp_academic_dataset_business.json'
# # input_file = 'data/yelp/yelp_academic_dataset_review.json'
# input_file = 'data/yelp/yelp_academic_dataset_user.json'

# # Path to the output file
# # output_file = 'processed_buainess.json'
# # output_file = 'processed_review.json'
# output_file = 'processed_user.json'

# # List to store the extracted data
# extracted_data = []

# def flatten_dict(d, parent_key='', sep='.'):
#     if d is None:
#         return {}
#     items = []
#     for k, v in d.items():
#         # if v is str and start with {, it is a dict
#         if isinstance(v, str) and v.startswith("{"):
#             v = eval(v)

#         new_key = f'{parent_key}{sep}{k}' if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)


# def process_dict(d, sep = '-'):
#     if d == {}:
#         return "-1"
#     flatten = flatten_dict(d)
#     # print(flatten)
#     output = ""
#     for k, v in flatten.items():
#         output += f"{k}{sep}{v},"

#     return output[:-1]

# # # Read the input JSON file
# # with open(input_file, 'r', encoding="utf8") as file:
# #     __i = 0
# #     # Read all the lines in the file
# #     try:
# #         for line in file:
# #             __i += 1
# #             # Parse the JSON data
# #             data = json.loads(line)
            
# #             # Extract the required fields
# #             extracted_data.append({
# #                 'business_id': data.get('business_id', -1),
# #                 'city': data.get('city', -1),
# #                 'stars': data.get('stars', -1),
# #                 'review_count': data.get('review_count', -1),
# #                 'is_open': data.get('is_open', -1),
# #                 'attributes': process_dict(data.get('attributes', {})),
# #                 'categories': data.get('categories', -1)
# #             })  
# #     except Exception as e:
# #         print(f"Error at line {__i}")
# #         print(e)


# # # Read the input JSON file
# # with open(input_file, 'r', encoding="utf8") as file:
# #     # read all line in the file
# #     for line in file:
# #         # parse the JSON data
# #         data = json.loads(line)
# #         # Extract the required fields
# #         # concat the key "useful" and its value with '.' and store in  "useful" as a string
# #         useful = data.get('useful', -1)
# #         # concat the key "funny" and its value with '.' and store in  "funny" as a string
# #         funny = data.get('funny', -1)
# #         cool = data.get('cool', -1)
# #         votes = f"useful.{useful},funny.{funny},cool.{cool}"
        
# #         extracted_data.append({
# #             'review_id': data.get('review_id', -1),
# #             'user_id': data.get('user_id', -1),
# #             'business_id': data.get('business_id', -1),
# #             'stars': data.get('stars', -1),
# #             'text': data.get('text', -1),
# #             'date': data.get('date', -1),
# #             # concattence the key "useful", "funny", "cool" with its value together with "."
# #             'votes': votes,
# #         })

# # # Read the input JSON file
# # with open(input_file, 'r', encoding="utf8") as file:
# #     # read all line in the file
# #     for line in file:
# #         # parse the JSON data
# #         data = json.loads(line)
# #         # Extract the required fields
# #         # concat the key "useful" and its value with '.' and store in  "useful" as a string
# #         yelping_since = data.get('yelping_since', -1)
# #         # concat the key "useful" and its value with '.' and store in  "useful" as a string
# #         useful = data.get('useful', -1)
# #         # concat the key "funny" and its value with '.' and store in  "funny" as a string
# #         funny = data.get('funny', -1)
# #         cool = data.get('cool', -1)
# #         votes = f"useful.{useful},funny.{funny},cool.{cool}"
# #         hot = data.get('compliment_hot',-1)
# #         more = data.get('compliment_more',-1)
# #         profile = data.get('compliment_profile',-1)
# #         cute = data.get('compliment_cute',-1)
# #         list = data.get('compliment_list',-1)
# #         note = data.get('compliment_note',-1)
# #         plain = data.get('compliment_plain',-1)
# #         cool = data.get('compliment_cool',-1)
# #         funny = data.get('compliment_funny',-1)
# #         writer = data.get('compliment_writer',-1)
# #         compliments = f"hot.{hot},more.{more},profile.{profile},cute.{cute},list.{list},note.{note},plain.{plain},cool.{cool},funny.{funny},writer.{writer}"

        
# #         extracted_data.append({
# #             'user_id': data.get('user_id', -1),
# #             'review_count': data.get('review_count', -1),
# #             'yelping_since': yelping_since,
            
# #             'fans': data.get('fans', -1),
# #             'average_stars': data.get('average_stars', -1),
# #             'compliments': compliments,
# #             'votes': votes,
# #             'friends': data.get('friends', -1)
# #         })


# # Read the input JSON file
# with open(input_file, 'r', encoding="utf8") as file:
#     __i = 0
#     # Read all the lines in the file
#     try:
#         for line in file:
#             __i += 1
#             # Parse the JSON data
#             data = json.loads(line)
            
#             # Extract the required fields
#             extracted_data.append({
#                 'business_id': data.get('business_id', -1),
#                 'city': data.get('city', -1),
#                 'stars': data.get('stars', -1),
#                 'review_count': data.get('review_count', -1),
#                 'is_open': data.get('is_open', -1),
#                 'attributes': process_dict(data.get('attributes', {})),
#                 'categories': data.get('categories', -1)
#             })  
#     except Exception as e:
#         print(f"Error at line {__i}")
#         print(e)

# # Write the extracted data to the output file line by line
# with open(output_file, 'w') as file:
#     for data in extracted_data:
#         file.write(json.dumps(data) + '\n')



# Path to the input JSON file
input_file = 'processed_review2.json'
# user_file = 'processed_user.json'
# Path to the output file
output_file = 'processed_review3.json'

# extract the data from the input file
# extract the user_id as the first column
# for the same user_id, process the key review_id, business_id by concating the key and its value with "." and stored as a string

# UserID = {}

# with open(user_file, 'r', encoding="utf8") as file:
#     for line in file:
#         data = json.loads(line)
        
#         # extract the user_id and friends
#         user_id = data.get('user_id', -1)
#         friends = data.get('friends', -1)
#         UserID[user_id] = friends
        

extracted_data = {}

with open(input_file, 'r', encoding="utf8") as file:
    # read all line in the file
    for line in file:
        # parse the JSON data
        data = json.loads(line)
        user_id = data.get('user_id', -1)
        # review_id = data.get('review_id', -1)
        business_id = data.get('business_id', -1)
        stars = data.get('stars', -1)
        
        if user_id not in extracted_data:
            extracted_data[user_id] = {
                'reviews': {
                    'FiveStar': [],
                    'FourStar': [],
                    'ThreeStar': [],
                    'TwoStar': [],
                    'OneStar': []
                }
            }
        
        if stars == 5:
            extracted_data[user_id]['reviews']['FiveStar'].append(business_id)
        elif stars == 4:
            extracted_data[user_id]['reviews']['FourStar'].append(business_id)
        elif stars == 3:
            extracted_data[user_id]['reviews']['ThreeStar'].append(business_id)
        elif stars == 2:
            extracted_data[user_id]['reviews']['TwoStar'].append(business_id)
        elif stars == 1:
            extracted_data[user_id]['reviews']['OneStar'].append(business_id)

# # Set the user_id as the super key, find the friends in the UserID dictionary and store in the extracted_data
# for user_id, data in extracted_data.items():
#     friends = UserID.get(user_id, -1)
#     extracted_data[user_id]['friends'] = friends

# Write the sorted list of dictionaries to the output file
with open(output_file, 'w') as file:
    for user_id, data in extracted_data.items():
        file.write(json.dumps({'user_id': user_id, **data}) + '\n')
        
