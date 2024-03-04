import os
import json
import re
import tracemalloc
import stat_lib as st
import time 
# from ./data/raw_data import files 

cat_name = st.CAT_NAME

raw_data_path = './data/raw_data'

# start tracking memory usage
tracemalloc.start()
__r_t = time.time()
for subcat in cat_name:
    print('='*64)
    print('Processing subcategory: {}'.format(subcat))
    print('-'*64)
    meta_path = os.path.join(raw_data_path, 'meta_{}.json'.format(subcat))
    review_path = os.path.join(raw_data_path, '{}.json'.format(subcat))

    
    # extract useful columns in files
    processed_meta = []
    # each line of the file is a json object
    with open(meta_path) as f:
        i = 0
        __len = len(f.readlines())
        __t = time.time()
        # read each line of the file, and convert the json object to a dictionary
        for line in f:
            item = json.loads(line)

            # if price is not in "$" format, set it to None, skip the checking if price is ''
            if 'price' in item and item['price'] != '' and item['price'].startswith("$"):
                # Remove symbols
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

            # remove unwanted characters from brand
            if 'brand' in item:
                item['brand'] = item['brand'].replace("by", "").replace("\n", "").replace(" ","").replace(".","").replace("*","").replace("(),","").replace("()","")
            else:
                item['brand'] = ''
            if (item['brand'] == '-') or (item['brand'] == "--") or (item['brand'] == '&'):
                item['brand'] = ''

            # only get the first 3 items in category
            if 'category' in item:
                item['category'] = item['category'][:2]
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

            i += 1
            # if one second has passed, print the progress
            if time.time() - __t > 1:
                __t = time.time()
                print('Processed {}/{} meta data'.format(i, __len))

    with open('./data/l1_data/meta_{}.json'.format(subcat), 'w') as f:
        # clear all the data in the file
        f.seek(0)
        for item in processed_meta:
            f.write(json.dumps(item) + '\n')
    
    print('Finished processing meta data for subcategory l1: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    __r_t = time.time()

    
    processed_review = []
    with open(review_path) as f:
        # review = [json.loads(line) for line in f]
        i = 0
        __len = len(f.readlines())
        __t = time.time()
        for line in f:
            __temp = json.loads(line)
            processed_review.append({
                'reviewerID': __temp['reviewerID'],
                'asin': __temp['asin'],
                'vote': __temp.get('vote', None),
                'overall': __temp['overall'],
                'unixReviewTime': __temp['unixReviewTime']
            })

            i += 1
            # if one second has passed, print the progress
            if time.time() - __t > 1:
                __t = time.time()
                print('Processed {}/{} review data'.format(i, __len))
        
    
    

    # show current memory usage of processed meta and review in mb, only show items that use more than 1 mb
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # for stat in top_stats:
    #     if stat.size / 10**6 > 1:
    #         print(stat)

    # save to json, don't compress

    
    with open('./data/l1_data/review_{}.json'.format(subcat), 'w') as f:
        # clear all the data in the file
        f.seek(0)
        for item in processed_review:
            f.write(json.dumps(item) + '\n')
    print('Finished processing review data for subcategory l1: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    print('-'*64)
    print('Finished processing subcategory l1: {}'.format(subcat))
    
    __r_t = time.time()
    print('-'*64)
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
    i = 0
    __len = len(processed_review)
    __t = time.time()
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
        
        i += 1
        # if one second has passed, print the progress
        if time.time() - __t > 1:
            __t = time.time()
            print('Merged {}/{} review data'.format(i, __len))

    del processed_review

    print('Finished merging review data for subcategory l2: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    print('-'*64)
    __r_t = time.time()

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
        f.seek(0)
        for item in merged:
            f.write(json.dumps(item) + '\n')
    
    with open('./data/l2_data/core_{}.json'.format(subcat), 'w') as f:
        f.seek(0)
        for item in core:
            f.write(json.dumps(item) + '\n')
    print('Finished saving merged and core data for subcategory l2: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    print('-'*64)
    print('Finished processing subcategory l2: {}'.format(subcat))
    
# # show current memory usage of processed meta and review in mb, only show items that use more than 1 mb
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
# for stat in top_stats:
#     if stat.size / 10**6 > 1:
#         print(stat)

# stop tracking memory usage
tracemalloc.stop()