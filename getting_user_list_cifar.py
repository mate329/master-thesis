import os

def get_user_list(data_directory):
    # List to hold names of users (subdirectory names)
    users = []

    # Check each item in the data directory
    for item in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item)
        # If the item is a directory, add it to the user list
        if os.path.isdir(item_path):
            users.append(item)

    return users

# Example usage:
data_directory = './spectrograms_entryActivity'  # Replace with the path to your dataset directory
user_list = get_user_list(data_directory)
print("User classes found:", user_list)
