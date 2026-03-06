"""
download.py - Download the Tiny Shakespeare dataset
====================================================

GOAL:
    Download the Tiny Shakespeare text file and save it as `input.txt` at the project root.

WHAT TO IMPLEMENT:
    1. Use `urllib.request` (or `requests`) to download the file from:
       https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    2. Save the downloaded content to `../input.txt` (project root).

    3. Print a confirmation message with the file size (number of characters).

HINTS:
    - The file is ~1.1MB of raw text (~1,115,394 characters).
    - Use `if __name__ == "__main__":` so this script can be run standalone.
    - Consider adding a check: skip download if the file already exists.

STRETCH:
    - Print a short preview of the first 200 characters after downloading.
"""
