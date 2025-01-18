#/home/woody/vlgm/vlgm103v/dataset/Data_Deschaintre18/trainBlended/
import os

def count_files_in_folder(folder_path):
    """
    Counts the number of files in a given folder.

    Parameters:
    folder_path (str): The path to the folder.

    Returns:
    int: Number of files in the folder.
    """
    try:
        # List all entries in the folder
        entries = os.listdir(folder_path)
        
        # Filter out only files
        files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
        
        # Return the number of files
        return len(files)
    except Exception as e:
        print(f"Error: {e}")
        return 0

# Default folder path
folder_path = "/home/woody/vlgm/vlgm103v/dataset/Data_Deschaintre18/trainSketch/"

# Count files in the folder
num_files = count_files_in_folder(folder_path)
print(f"Number of files in the folder: {num_files}")
