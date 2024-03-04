import os
import json
import re
import tracemalloc
import stat_lib as st
import time 
# from ./data/raw_data import files 

cat_name = st.CAT_NAME

raw_data_path = './data/raw_data'
INTERVAL = 1 # print progress every 1 second

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
        with open('./data/l1_data/meta_{}.json'.format(subcat), 'w') as d:
            i = 0
            __len = len(f.readlines())
            __t = time.time()
            f.seek(0)
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
                
                __temp = {
                    'asin': item['asin'],
                    'price': item.get('price', None),
                    'also_view': item.get('also_view', None),
                    'also_buy': item.get('also_buy', None),
                    'rank': item.get('rank', None),
                    'brand': item.get('brand', None),
                    'category': item.get('category', None)
                }

                
                d.write(json.dumps(__temp) + '\n')
                del __temp
                i += 1
                # if one second has passed, print the progress
                if time.time() - __t > INTERVAL or i == __len:
                    __t = time.time()
                    print('Processed {}/{} ({:.2f}%) meta data'.format(i, __len, i/ __len * 100))

    
    
    print('Finished processing meta data for subcategory l1: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    __r_t = time.time()
    del processed_meta
    
    processed_review = []
    with open(review_path) as f:
        with open('./data/l1_data/review_{}.json'.format(subcat), 'w') as d:
            # review = [json.loads(line) for line in f]
            i = 0
            __len = len(f.readlines())
            __t = time.time()
            f.seek(0)
            for line in f:
                item = json.loads(line)
                __temp = {
                    'reviewerID': item['reviewerID'],
                    'asin': item['asin'],
                    'vote': item.get('vote', None),
                    'overall': item['overall'],
                    'unixReviewTime': item['unixReviewTime']
                }

                d.write(json.dumps(__temp) + '\n')    
                del __temp
                i += 1
                # if one second has passed, print the progress
                if time.time() - __t > INTERVAL or i == __len:
                    __t = time.time()
                    print('Processed {}/{} ({:.2f}%) review data'.format(i, __len, i/ __len * 100))
        
    

    # show current memory usage of processed meta and review in mb, only show items that use more than 1 mb
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # for stat in top_stats:
    #     if stat.size / 10**6 > 1:
    #         print(stat)

    # save to json, don't compress

    
    
            
    print('Finished processing review data for subcategory l1: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    print('-'*64)
    print('Finished processing subcategory l1: {}'.format(subcat))
    del processed_review
    __r_t = time.time()
    print('-'*64)
    # do l2 processing, merge meta and review data
    print('Processing subcategory l2: {}'.format(subcat))

    # create a new data
    merged = []
    merged_asin = {}
    with open('./data/l1_data/meta_{}.json'.format(subcat)) as f:
        i = 0
        __len = len(f.readlines())
        __t = time.time()
        f.seek(0)
        # read each line of the file, and convert the json object to a dictionary
        for line in f:
            item = json.loads(line)
            
            
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
            merged_asin[item['asin']] = i
            i += 1
            # if one second has passed, print the progress
            if time.time() - __t > INTERVAL or i == __len:
                __t = time.time()
                print('Processed {}/{} ({:.2f}%)  l1 meta data'.format(i, __len, i/ __len * 100))
    

    with open('./data/l1_data/review_{}.json'.format(subcat)) as f:
        i = 0
        __len = len(f.readlines())
        __t = time.time()
        f.seek(0)
        for line in f:
            __temp = json.loads(line)
            asin = __temp['asin']
            # find the corresponding meta data
            # get the index of the asin in the merged data
            if asin not in merged_asin:
                continue
            index = merged_asin[asin]
            # append the review to the merged data
            merged[index]['reviews'].append({
                'reviewerID': __temp['reviewerID'],
                'vote': __temp['vote'],
                'overall': __temp['overall'],
                'unixReviewTime': __temp['unixReviewTime']
            })
            
            i += 1
            # if one second has passed, print the progress
            if time.time() - __t > INTERVAL or i == __len:
                __t = time.time()
                print('Merged {}/{} ({:.2f}%) review data'.format(i, __len, i/ __len * 100))

    print('Finished merging review data for subcategory l2: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    print('-'*64)
    __r_t = time.time()

    # calculate the average rating for each product
    __len = len(merged)
    i = 0
    __t = time.time()
    for item in merged:
        reviews = item['reviews']
        if len(reviews) == 0:
            item['avg_rating'] = None
        else:
            item['avg_rating'] = sum([r['overall'] for r in reviews]) / len(reviews)

        i += 1
        # if one second has passed, print the progress
        if time.time() - __t > INTERVAL or i == __len:
            __t = time.time()
            print('Processed {}/{} ({:.2f}%) average rating'.format(i, __len, i/ __len * 100))
        
    print('Finished calculating average rating for subcategory l2: {}'.format(subcat))
    print('Time taken: {:.3f}s'.format(time.time() - __r_t))
    print('-'*64)
    __r_t = time.time()

        
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