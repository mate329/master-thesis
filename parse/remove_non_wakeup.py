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

    # Find the index of the 'angularSpeedX' column
    if data:
        header = data[0]
        try:
            angular_speed_index = header.index('angularSpeedX')
        except ValueError:
            print(f"'angularSpeedX' column not found in {file}")
            return

        # Process each row
        processed_data = [header]
        for row in data[1:]:
            if not is_number(row[angular_speed_index]):
                # If 'angularSpeedX' is not a number, remove it and shift subsequent values to the left
                row.pop(angular_speed_index)
                row.append('')  # Append an empty string to keep the row length consistent
            processed_data.append(row)

        # Write processed data back to the file
        with open(file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(processed_data)

def process_files_recursive(start_path):
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_file(file_path)
                print(f"Processed {file_path}")

# Starting from the current directory
process_files_recursive('.')
