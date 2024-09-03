import os

def delete_results_txt(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'results.txt':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Specify the directory to start from
start_directory = "./bankAppData"  # replace with your directory path
delete_results_txt(start_directory)
