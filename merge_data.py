import os
import json
import re
import tracemalloc
import stat_lib as st
# from ./data/raw_data import files 

cat_name = st.CAT_NAME

raw_data_path = './data/raw_data'

# start tracking memory usage
tracemalloc.start()

for subcat in cat_name:
    print('='*64)
    print('Processing subcategory: {}'.format(subcat))
    meta_path = os.path.join(raw_data_path, 'meta_{}.json'.format(subcat))
    review_path = os.path.join(raw_data_path, '{}.json'.format(subcat))

    

    # each line of the file is a json object
    with open(meta_path) as f:
        # read each line of the file, and convert the json object to a dictionary
        meta = [json.loads(line) for line in f]

    with open(review_path) as f:
        review = [json.loads(line) for line in f]

    

    # extract useful columns in files
    processed_meta = []
    for item in meta:
        # if price is not in "$" format, set it to None, skip the checking if price is ''
        if 'price' in item and item['price'] != '':

            # Check if the price does not start with "$"
            if not item['price'].startswith("$"):
                item['price'] = ["-1", "-1"] # Set to ["-1", "-1"] if it doesn't start with "$"
                continue # Skip the rest of the loop for this item

            # Remove the "$" symbol
            price = re.sub(r"[^\d.]", "", item['price'].replace("$", "").replace(",", "").replace(" ",""))
            if "-" in price:
                # It's a range, split and process
                lower, upper = map(float, price.split("-"))
                mean = (lower + upper) / 2
                diff = upper - mean
                item['price'] = [str(mean), str(diff)]
            else:
                # It's an actual amount
                item['price'] = [price, '0']
        else:
            item['price'] = ["-1", "-1"] # Set to [0, 0] if price is not present or empty

        # if brand name contain "by\n    \n    ", remove it
        item['brand'] = item['brand'].replace("by", "").replace("\n", "").replace(" ","").replace(".","").replace("*","").replace("(),","").replace("()","")
        if (item['brand'] == '-') or (item['brand'] == "--") or (item['brand'] == '&'):
            item['brand'] = ''

        # only get the first 3 items in category
        if 'category' in item:
            item['category'] = item['category'][:3]
            # Check if "</span></span></span>" exists in category and remove it
            item['category'] = [category.replace("</span></span></span>", "") for category in item['category']]
            # Remove empty elements from the first 3 elements in 'category'
            item['category'] = list(filter(None, item['category']))
        
        processed_meta.append({
            'asin': item['asin'],
            'price': item.get('price', None),
            'also_view': item.get('also_view', None),
            'also_buy': item.get('also_buy', None),
            'rank': item.get('rank', None),
            'brand': item.get('brand', None),
            'category': item.get('category', None)
        })

    processed_review = []
    for item in review:
        processed_review.append({
            'reviewerID': item['reviewerID'],
            'asin': item['asin'],
            'vote': item.get('vote', None),
            'overall': item['overall'],
            'unixReviewTime': item['unixReviewTime']
        })
    
    

    # show current memory usage of processed meta and review in mb, only show items that use more than 1 mb
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # for stat in top_stats:
    #     if stat.size / 10**6 > 1:
    #         print(stat)

    # save to json, don't compress
    with open('./data/l1_data/meta_{}.json'.format(subcat), 'w') as f:
        for item in processed_meta:
            f.write(json.dumps(item) + '\n')
    
    with open('./data/l1_data/review_{}.json'.format(subcat), 'w') as f:
        for item in processed_review:
            f.write(json.dumps(item) + '\n')

    print('Finished processing subcategory l1: {}'.format(subcat))

    # clean up memory
    del meta
    del review

    # do l2 processing, merge meta and review data
    print('Processing subcategory l2: {}'.format(subcat))

    # create a new data
    merged = []
    for item in processed_meta:
        merged.append({
            'asin': item['asin'],
            'price': item['price'],
            'also_view': item['also_view'],
            'also_buy': item['also_buy'],
            'rank': item['rank'],
            'brand': item['brand'],
            'category': item['category'],
            'reviews': []
        })
    
        
    del processed_meta

    # use the processed meta and review data to reduce memory usage
    # merged_asin = [item['asin'] for item in merged]
    merged_asin = {item['asin']: i for i, item in enumerate(merged)}
    for item in processed_review:
        asin = item['asin']
        # find the corresponding meta data
        # get the index of the asin in the merged data
        if asin not in merged_asin:
            continue
        index = merged_asin[asin]
        # append the review to the merged data
        merged[index]['reviews'].append({
            'reviewerID': item['reviewerID'],
            'vote': item['vote'],
            'overall': item['overall'],
            'unixReviewTime': item['unixReviewTime']
        })
        pass

    del processed_review

    # calculate the average rating for each product
    for item in merged:
        reviews = item['reviews']
        if len(reviews) == 0:
            item['avg_rating'] = None
        else:
            item['avg_rating'] = sum([r['overall'] for r in reviews]) / len(reviews)

        
    core = []
    for item in merged:
        core.append({
            'asin': item['asin'],
            'price': item['price'],
            'avg_rating': item['avg_rating'],
            'rank': item['rank'],
            'brand': item['brand'],
            'category': item['category']
        })

    # save to json, don't compress
    with open('./data/l2_data/merged_{}.json'.format(subcat), 'w') as f:
        for item in merged:
            f.write(json.dumps(item) + '\n')
    
    with open('./data/l2_data/core_{}.json'.format(subcat), 'w') as f:
        for item in core:
            f.write(json.dumps(item) + '\n')

    

# # show current memory usage of processed meta and review in mb, only show items that use more than 1 mb
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
# for stat in top_stats:
#     if stat.size / 10**6 > 1:
#         print(stat)

# stop tracking memory usage
tracemalloc.stop()