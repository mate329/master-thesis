import csv
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_file(file):
    with open(file, newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Process file only if it has data and a header
    if len(data) > 1 and len(data[0]) > 0:
        # Find the index of the 'sensorType' column
        try:
            sensor_type_index = data[0].index('sensorType')
        except ValueError:
            print(f"'sensorType' column not found in {file}")
            return

        # Process each row
        processed_data = [data[0]]
        for row in data[1:]:
            if is_number(row[sensor_type_index]):
                # Move all data one column to the right
                row.insert(0, '')
            processed_data.append(row)

        # Write processed data back to the file
        with open(file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(processed_data)

def process_files_recursive(start_path):
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith('.csv') and 'enterPINactivity' in file:
                file_path = os.path.join(root, file)
                process_file(file_path)
                print(f"Processed {file_path}")

# Starting from the current directory
process_files_recursive('../bankAppDataWithPlot')
