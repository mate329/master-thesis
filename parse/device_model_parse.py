import os
import csv
import shutil

def convert_txt_to_csv_and_move(directory, target_root):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if "deviceModel" in filename and filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                # Construct CSV file name based on the TXT file name
                csv_file_name = os.path.splitext(filename)[0] + '.csv'
                csv_file_path = os.path.join(root, csv_file_name)

                with open(file_path, 'r') as file:
                    data = file.read()
                    parts = data.split()
                    if len(parts) >= 4:
                        # Extracting screenResolutionX and screenResolutionY
                        screenResolutionX = parts[2]
                        screenResolutionY = parts[3]

                        # Write to CSV
                        with open(csv_file_path, 'w', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerow(['screenResolutionX', 'screenResolutionY'])
                            csvwriter.writerow([screenResolutionX, screenResolutionY])

                # Create the same directory structure in the target root
                relative_path = os.path.relpath(root, directory)
                target_directory = os.path.join(target_root, relative_path)
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)

                # Move the CSV file to the target directory
                target_csv_path = os.path.join(target_directory, csv_file_name)
                shutil.move(csv_file_path, target_csv_path)
                print(f"Moved CSV file to: {target_csv_path}")

# Replace these with the paths to the source and target directories
source_directory = "../userInfoResolution"
target_directory = "../bankAppData"
convert_txt_to_csv_and_move(source_directory, target_directory)
