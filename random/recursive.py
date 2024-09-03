import csv
import os
import fnmatch

def shift_data_right(file, start_column, fill_value):
    with open(file, newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Find the index of the start_column
    header = data[0]
    try:
        start_index = header.index(start_column)
    except ValueError:
        print(f"Column '{start_column}' not found in {file}")
        return

    # Shift data one column to the right from start_index onwards for all rows except the header
    for row in data[1:]:
        row.insert(start_index, fill_value)

    with open(file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

def process_files_recursive(start_path):
    for root, dirs, files in os.walk(start_path):
        for filename in fnmatch.filter(files, '*onScroll*.csv'):
            file_path = os.path.join(root, filename)
            shift_data_right(file_path, 'e2Size', '0')
            print(f"Processed and updated {file_path}")

# Starting from the current directory
process_files_recursive('.')

