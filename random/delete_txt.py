import os

# Function to delete .txt files in a given directory
def delete_txt_files(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Path to the directory where .txt files should be deleted
directory_path = './bankAppData'  # Replace with your directory path

# Check if directory exists
if not os.path.exists(directory_path):
    print(f"Directory does not exist: {directory_path}")
else:
    print(f"Deleting .txt files in directory: {directory_path}")
    delete_txt_files(directory_path)
    print("All .txt files have been deleted.")

