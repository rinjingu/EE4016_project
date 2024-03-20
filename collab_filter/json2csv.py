import json, csv, sys, os

# Check if the file name is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python json2csv.py <json_file>")
    sys.exit(1)

# Get the JSON file name from the command-line argument
json_file = sys.argv[1]

# Get the base name of the JSON file without the extension
output_file = os.path.splitext(os.path.basename(json_file))[0] + ".csv"

# Open the JSON file for reading
try:
    with open(json_file, "r") as file:
        # Create a list to store the data
        data = []

        # Read each line in the JSON file
        for line in file:
            try:
                # Load the JSON data from the line
                item = json.loads(line)

                # Extract the desired columns
                asin = item.get("asin", "")
                avg_rating = item.get("avg_rating", 0.0)
                brand = item.get("brand", 0.0)
                activeness = item.get("activeness", 0.0)

                # Add the data to the list
                data.append({"asin": asin, "avg_rating": avg_rating, "brand": brand, "activeness": activeness})
            except json.JSONDecodeError:
                # Ignore lines that are not valid JSON
                pass
except FileNotFoundError:
    print(f"Error: File '{json_file}' not found.")
    sys.exit(1)

# Open the CSV file for writing
with open(output_file, "w", newline="") as csv_file:
    # Create a CSV writer object
    writer = csv.DictWriter(csv_file, fieldnames=["asin", "brand", "avg_rating", "activeness"])

    # Write the header row
    writer.writeheader()

    # Write the data rows
    for item in data:
        writer.writerow(item)

print(f"CSV file '{output_file}' created successfully!")
