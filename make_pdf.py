"""
create the pdf file from the json file provided by clapnq github repo :
https://github.com/primeqa/clapnq/blob/main/README.md
"""

import json
from fpdf import FPDF


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


def list_to_pdf(data_list, pdf_path):
    """
    Converts a list of strings into a PDF file.

    Parameters:
        data_list (list): The list of strings to include in the PDF.
        pdf_path (str): The path to save the generated PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in data_list:
        line = line.encode("latin-1", "replace").decode("latin-1")
        # Split the line into chunks that fit the width of the page
        max_width = pdf.w - 2 * pdf.l_margin
        words = line.split()
        current_line = ""

        for word in words:
            if pdf.get_string_width(current_line + word + " ") <= max_width:
                current_line += word + " "
            else:
                pdf.cell(0, 10, txt=current_line.strip(), ln=True)
                current_line = word + " "

        # Add the remaining text
        if current_line:
            pdf.cell(0, 10, txt=current_line.strip(), ln=True)

    try:
        pdf.output(pdf_path)
        print(f"PDF successfully created at {pdf_path}")
    except Exception as e:
        print(f"Error creating PDF: {e}")


# Read clap nq corpus:
file_path = "clapnq_train_answerable.jsonl"
records = read_jsonl(file_path)
corpus = []
for record in records:
    # passages
    for text in record["passages"][0]["sentences"]:
        corpus.append(text)
        print(text)

# make the pdf file
list_to_pdf(corpus, "clapnq_corpus.pdf")


# print(corpus)
# question
# print(record['input'])

# answer
# print(record['output'][0]['answer'])
