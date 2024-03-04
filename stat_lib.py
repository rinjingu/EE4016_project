import math
import json

CAT_NAME = [
    'Gift_Cards',
    "All_Beauty",
    "Appliances",
    "Digital_Music",
    "CDs_and_Vinyl",
    "Automotive"

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
    activeness = 0

    # iterate through the reviews
    for review in reviews:
        # get the time of the review
        r_time = review['unixReviewTime']
        # calculate the k value
        k = math.log(DATA_TIME - r_time + 1)
        # add the k value to the activeness
        activeness += k
        pass

    return activeness
    

def label_string(items, key):
    labels = {}
    i = 0
    for item in items:
        if item[key] not in labels:
            labels[item[key]] = i
            i += 1
    return labels

def label_category(items):
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
    
    return labels

def map_relation(items, significantness=1, buy_effect=1, view_effect=1):
    # create a dictionary to store the relation
    relation = {}
    # iterate through the items
    for item in items:
        # get the asin of the item
        asin = item['asin']
        map_ = {}
        # iterate through the also_buy and also_view
        for a_item in item['also_buy']:
            if a_item in map_:
                map_[a_item] += buy_effect
            else:
                map_[a_item] = buy_effect

        for a_item in item['also_view']:
            if a_item in map_:
                map_[a_item] += view_effect
            else:
                map_[a_item] = view_effect

        # add the relation to the dictionary
        relation[asin] = map_ 



    return relation
