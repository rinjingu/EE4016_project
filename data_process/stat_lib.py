import math
import json
import time

CAT_NAME = [
    #  "Grocery_and_Gourmet_Food",
     "Industrial_and_Scientific"
    # "Magazine_Subscriptions",
    # "Musical_Instruments",
    # "Office_Products",
    # "Patio_Lawn_and_Garden",
    # "Pet_Supplies",
    # "Prime_Pantry",
    # "Software",
    # "Sports_and_Outdoors",
    # "Tools_and_Home_Improvement",
    # "Toys_and_Games",
    # "Video_Games"   

    # "Books"

    # "Gift_Cards",
    # "AMAZON_FASHION",
    # "Arts_Crafts_and_Sewing",
    # "All_Beauty",
    # "Appliances",
    # "Digital_Music",
    # "CDs_and_Vinyl",
    # "Automotive",
    # "Clothing_Shoes_and_Jewelry",
    # "Cell_Phones_and_Accessories",
    # "Electronics",
    # "Home_and_Kitchen",
    # "Movies_and_TV",
    # "Luxury_Beauty",
    # "Kindle_Store"
]

def open_file(file_name):
    try:
        with open(file_name) as f:
            # read each line of the file, and convert the json object to a dictionary
            data = [json.loads(line) for line in f]

        return data
    except Exception as e:
        raise e
    

# Function to calculate the review activeness of a product
DATA_TIME = 1546300800
def review_activeness(reviews):
    __t = time.time()
    __i = 0
    __len = len(reviews)
    activeness = 0

    # iterate through the reviews
    for review in reviews:
        # get the time of the review
        r_time = review['unixReviewTime']
        # calculate the k value
        k = math.log(DATA_TIME - r_time + 1)
        # add the k value to the activeness
        activeness += k
        # __i += 1
        if time.time() - __t > 1 or __i == __len:
            __t = time.time()
            print('Processed {}/{} ({:.2f}%) review activeness'.format(__i, __len, __i/ __len * 100))
        pass

    return activeness
    

def label_string(items, key):
    __t = time.time()
    __len = len(items)
    __i = 0
    labels = {}
    i = 0
    for item in items:
        if item[key] not in labels:
            labels[item[key]] = i
            i += 1
            
        __i += 1
        if time.time() - __t > 1 or __i == __len:
            __t = time.time()
            print('Processed {}/{} ({:.2f}%) label string'.format(__i, __len, __i/ __len * 100))
    return labels

def label_category(items):
    __t = time.time()
    __len = len(items)
    __i = 0
    labels = {}
    i = 0
    
    # iterate through the items
    for item in items:
        # get the category of the item
        cat = item['category']

        # if the category is a list, iterate through the list
        if type(cat) == list:
            for c in cat:
                if c not in labels:
                    labels[c] = i
                    i += 1
        else:
            if cat not in labels:
                labels[cat] = i
                i += 1
                
        __i += 1
        if time.time() - __t > 1 or __i == __len:
            __t = time.time()
            print('Processed {}/{} ({:.2f}%) label category'.format(__i, __len, __i/ __len * 100))
    
    return labels

def label_asin(items):
    __t = time.time()
    __len = len(items)
    __i = 0
    labels = {}
    i = 0
    
    # iterate through the items
    for item in items:
        # get the asin of the item
        asin = item['asin']
        # add the asin to the dictionary
        if asin not in labels:
            labels[asin] = i
            i += 1

        also_view = item['also_view']
        also_buy = item['also_buy']
        
        also = also_view + also_buy
        for a in also:
            if a not in labels:
                labels[a] = i
                i += 1
        
        __i += 1
        if time.time() - __t > 1 or __i == __len:
            __t = time.time()
            print('Processed {}/{} ({:.2f}%) label asin'.format(__i, __len, __i/ __len * 100))
    
    return labels

def map_relation(item, asins, significantness=1, buy_effect=1, view_effect=1, length_asin=0):
    if length_asin == 0:
        raise ValueError('length_asin cannot be 0')
    
    # create a dictionary to store the relation
    relation = {}
    
    # get the asin of the item
    asin = item['asin']
    map_ = {}
    # iterate through the also_buy and also_view
    for a_item in item['also_buy']:
        a_item = asins[a_item]
        if a_item in map_:
            map_[a_item] += buy_effect
        else:
            map_[a_item] = buy_effect

    for a_item in item['also_view']:
        a_item = asins[a_item]
        if a_item in map_:
            map_[a_item] += view_effect
        else:
            map_[a_item] = view_effect

    
    # add the relation to the dictionary
    relation = map_ 
    

    return relation
