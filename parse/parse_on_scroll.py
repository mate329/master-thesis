import csv
import glob

path = 'bankAppData/Adriano_Milanovic_1683369062627'

def shift_data_right(file, start_column, fill_value):
    with open(path + file, newline='') as infile:
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

def process_files():
    # Find all CSV files containing "enterPINactivity" in the current directory
    files = glob.glob('*onScroll*.csv')

    for file in files:
        shift_data_right(file, 'e2Size', '0')
        print(f"Processed and updated {file}")

# Run the function to process the files
process_files()
