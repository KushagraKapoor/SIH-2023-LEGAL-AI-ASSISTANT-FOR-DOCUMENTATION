import pyautogui as gui 
import webbrowser
import os
import time

def select(contents):
    webbrowser.open("https://chat.openai.com/c/72fbe740-d8f1-4ac5-9282-38c46a28fb6a")
    time.sleep(6)
    gui.hotkey('tab',interval = 0.2)
    gui.write(contents+"\n\n get me refined keywords",interval = 0.01)



# Specify the directory path where your text files are located
folder_path = '/Volumes/NO NAME/Smarthac/dataset/IN-Abs/train-data/summary'

# List all files in the directory
file_list = os.listdir(folder_path)

# Filter only the text files (you can adjust the condition as needed)
text_files = [file for file in file_list if file.endswith('.txt')]

# Loop through each text file and open it
for file_name in text_files:
    file_path = os.path.join(folder_path, file_name)
    
    # Open the file for reading
    with open(file_path, 'r') as file:
        # Read and process the contents of the file
        file_contents = file.read()
        select(file_contents) 
        
        # You can perform any processing or analysis on file_contents here
        # For example, print the content of each file
        #print(f'Contents of {file_name}:\n{file_contents}')
