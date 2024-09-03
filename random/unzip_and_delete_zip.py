import os
import zipfile

def unzip_to_named_folders_and_delete_zip_files(directory):
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Folder name is the file name without the '.zip' extension
            folder_name = os.path.splitext(filename)[0]
            # Full path for the new folder
            folder_path = os.path.join(directory, folder_name)

            # Create a folder with the same name as the zip file
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder {folder_path}")

            # Unzip the file into the newly created folder
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
                print(f"Unzipped {file_path} into {folder_path}")

            # Delete the zip file
            os.remove(file_path)
            print(f"Deleted {file_path}")

# Replace this with the path to the directory containing your zip files
directory = "./userInfoResolution"
unzip_to_named_folders_and_delete_zip_files(directory)
