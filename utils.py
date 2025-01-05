import json


def read_jsonl(file_path):
    """
    Reads a JSONL (JSON Lines) file and returns a list of dictionaries.

    Parameters:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, each representing a line in the file.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data
