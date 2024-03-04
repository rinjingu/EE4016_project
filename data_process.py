import os
import json
import tracemalloc
import stat_lib as st

cat_name = st.CAT_NAME

data_path = './data'
l2_data_path = os.path.join(data_path, 'l2_data')
l3_data_path = os.path.join(data_path, 'l3_data')
# start tracking memory usage
tracemalloc.start()

for subcat in cat_name:
    try:
        print('='*64)
        print('Processing subcategory: {}'.format(subcat))
        merged_path = os.path.join(
            l2_data_path, 'merged_{}.json'.format(subcat))
        core_path = os.path.join(
            l2_data_path, 'core_{}.json'.format(subcat))

        data = st.open_file(merged_path)
        core = st.open_file(core_path)

        # raise error if length of data and core is not equal
        if len(data) != len(core):
            raise ValueError('Length of data and core is not equal')

        brands = st.label_string(core, 'brand')
        subcats = st.label_category(core)
        # dump the brand label to a file, for future reference, each label is a key-value pair as a new line
        with open(os.path.join(l3_data_path, 'brand_label_{}.yaml'.format(subcat)), 'w') as f:
            for k, v in brands.items():
                f.write('{}: {}\n'.format(k, v))

        # dump the subcategory label to a file, for future reference, each label is a key-value pair as a new line
        with open(os.path.join(l3_data_path, 'subcat_label_{}.yaml'.format(subcat)), 'w') as f:
            for k, v in subcats.items():
                f.write('{}: {}\n'.format(k, v))

        for i in range(len(core)):
            # calculate the review activeness of each product
            core[i]['activeness'] = st.review_activeness(data[i]['reviews'])
            # round the activeness to 4 decimal places
            core[i]['activeness'] = round(core[i]['activeness'], 4)

            # convert the brand to a label of integer
            core[i]['brand'] = brands[core[i]['brand']]

            # process the rank of the product
            # if the rank is a array, take the first element
            if core[i]['rank'] == []:
                core[i]['rank'] = ''
            if type(core[i]['rank']) == list:
                core[i]['rank'] = core[i]['rank'][0]
            # extract the number from the string, the number is the first word in the string
            if core[i]['rank'] != '':
                core[i]['rank'] = core[i]['rank'].split()[0]
                # remove non-number characters from the rank
                core[i]['rank'] = int(
                    ''.join(filter(str.isdigit, core[i]['rank'])))
            else:
                core[i]['rank'] = -1

            # process the category of the product
            # if the category is a list, process each element in the list
            if type(core[i]['category']) == list:
                for j in range(len(core[i]['category'])):
                    core[i]['category'][j] = subcats[core[i]['category'][j]]
            else:
                core[i]['category'] = subcats[core[i]['category']]

            # limit the digit of the average rating to 2
            if core[i]['avg_rating'] is None or core[i]['avg_rating'] == '':
                core[i]['avg_rating'] = 0
            core[i]['avg_rating'] = round(core[i]['avg_rating'], 2)

            pass

        # write the processed data to a new file
        with open(os.path.join(l3_data_path, 'processed_{}.json'.format(subcat)), 'w') as f:
            for item in core:
                f.write(json.dumps(item) + '\n')

    except Exception as e:
        print('Error processing subcategory: {}'.format(subcat))
        # print(e)
        # continue
        raise e


# stop tracking memory usage
tracemalloc.stop()
