import csv
import os
import fnmatch

def shift_data_right(file):
    with open(file, newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Shift data one column to the right for all rows except the header
    for i in range(1, len(data)):
        data[i].insert(0, '')

    with open(file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

def process_files_recursive(start_path):
    for root, dirs, files in os.walk(start_path):
        for filename in fnmatch.filter(files, '*entryActivity*.csv'): # entryActivity enterPINactivity onScroll
            file_path = os.path.join(root, filename)
            shift_data_right(file_path)
            print(f"Processed and updated {file_path}")

        for filename in fnmatch.filter(files, '*enterPINactivity*.csv'): # entryActivity enterPINactivity onScroll
            file_path = os.path.join(root, filename)
            shift_data_right(file_path)
            print(f"Processed and updated {file_path}")

# Starting from the current directory
process_files_recursive('./bankAppData/Andrea_Morsi_1683316892728')
