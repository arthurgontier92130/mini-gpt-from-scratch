"""
download.py - Download the Tiny Shakespeare dataset
====================================================

GOAL:
    Download the Tiny Shakespeare text file and save it as `input.txt` at the project root.

"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, "input.txt")

def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(FILE_PATH):
        print("Dataset already downloaded")
        return
    print("Downloading Tiny Shakespeare dataset")
    urllib.request.urlretrieve(URL, FILE_PATH)
    print("Download completed.")

if __name__ == "__main__":
    download()
    
