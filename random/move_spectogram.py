import os
import shutil

def move_spectrograms(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through all files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if 'entryActivity' is in the filename
            if 'enterPINactivity' in file and file.startswith('spectrogram_'):
                # Extract the user information and timestamp
                # Filename format: 'spectrogram_angularSpeedY_5_entryActivity-Antonio_Stipanović_1683624060999.png'
                parts = file.split('enterPINactivity-')
                if len(parts) > 1:
                    user_info = parts[1].rsplit('.', 1)[0]  # 'Antonio_Stipanović_1683624060999'
                    user_name, timestamp = user_info.rsplit('_', 1)

                    # Create a directory for the user if it doesn't exist
                    user_dir = os.path.join(target_dir, user_name)
                    os.makedirs(user_dir, exist_ok=True)

                    # Define source and destination paths
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(user_dir, f"{timestamp}_{file}")

                    # Move the file
                    shutil.move(src_path, dest_path)
                    print(f"Moved {file} to {dest_path}")

def main():
    source_dir = '../bankAppDataWithPlot'
    target_dir = '../spectograms/spectrograms_enterPINactivity'
    move_spectrograms(source_dir, target_dir)

if __name__ == "__main__":
    main()
