
import os

folder_path = "projekt/base"
counter = 1

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        old_file_path = os.path.join(folder_path, filename)
        new_file_name = str(counter) + ".jpg"
        new_file_path = os.path.join(folder_path, new_file_name)

        try:
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")
            
        except FileNotFoundError:
            print(f"File {old_file_path} not found.")
        except FileExistsError:
            print(f"File {new_file_path} already exists.")
        counter += 1
print("Renaming completed.")