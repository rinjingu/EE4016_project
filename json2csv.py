import json, csv, sys, os

def user_csv(json_file, output_file):
    # Open the JSON file for reading
    try:
        with open(json_file, "r") as file:
            data = []
            for line in file:
                try:
                    item = json.loads(line)
                    reviewerID = item.get("reviewerID", "")
                    asin = item.get("asin", "")
                    overall = item.get("overall", 0.0)
                    unixReviewTime = item.get("unixReviewTime", 0.0)

                    # user_id, product_id, rating, review_time
                    data.append({"user_id": reviewerID, "product_id": asin, "rating": overall, "review_time": unixReviewTime})
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["user_id", "product_id", "rating", "review_time"])
        writer.writeheader()
        for item in data:
            writer.writerow(item)
    print(f"CSV file '{output_file}' created successfully!")

def item_csv(json_file, output_file):
    try:
        with open(json_file, "r") as file:
            data = []
            for line in file:
                try:
                    item = json.loads(line)
                    asin = item.get("asin", "")
                    brand = item.get("brand", "")

                    category_list = item.get("category", [])
                    if len(category_list) > 0:
                        category_index0 = category_list[0] 
                    else:
                        raise ValueError("Condition is false. An error occurred.")
                    if len(category_list) > 1:
                        category_index1 = category_list[1] 
                    else:
                        raise ValueError("Condition is false. An error occurred.")
                    
                    price_list = item.get("price", [])
                    if float(price_list[0]) > 0:
                        price_index0 = price_list[0]  
                    else:
                        raise ValueError("Condition is false. An error occurred.")
                    # price_index1 = price_list[1] if len(price_list) == 1 else None

                    data.append({"product_id": asin, "type_of_product": category_index0, "category": category_index1, "brand": brand, "price": price_index0})
                except json.JSONDecodeError:
                    pass
                except ValueError:
                    pass
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["product_id", "type_of_product", "category", "brand", "price"])
        writer.writeheader()
        for item in data:
            writer.writerow(item)

    print(f"CSV file '{output_file}' created successfully!")

if len(sys.argv) < 2:
    print("Usage: python json2csv.py <json_file>")
    sys.exit(1)

json_file = sys.argv[1]
# output_file = os.path.splitext(os.path.basename(json_file))[0] + ".csv"

file_type = json_file.split("_")[0]
if file_type == "review":
    output_file = "user_base_data.csv"
    user_csv(json_file, output_file)
elif file_type == "processed":
    output_file = "item_base_data.csv"
    item_csv(json_file, output_file)
else:
    print("Input file error.")