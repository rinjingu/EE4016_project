import json
import pandas as pd



# Function to flatten JSON
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            for i, a in enumerate(x):
                if type(a) is dict:
                    flatten(a, name + str(i) + '_')
                else:
                    out[name + str(i)] = a
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# Read the JSON file
flattened_data = []
with open('yelp/yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
    for _, line in enumerate(f):
        data = json.loads(line)
        flattened_data.append(flatten_json(data))

# Convert the flattened data to a DataFrame
df = pd.DataFrame(flattened_data)

# Fill missing values
df.fillna(0, inplace=True)
df['categories'] = df['categories'].str.split(', ')
# Convert the categories to separate columns
df_categories = df['categories'].apply(pd.Series)
df_categories = pd.get_dummies(df_categories.stack()).groupby(level=0).sum()

# Drop the original categories column and add the new ones
df = pd.concat([df.drop('categories', axis=1), df_categories], axis=1)

print(df.columns)

# Write the processed data to a new JSON file
df.to_json('for_item.json', orient='records', lines=True)

