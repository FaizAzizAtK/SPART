import pandas as pd
import os
import kagglehub

def generate_raw_text_and_format(dataset):
    raw_texts = []  # Stores the full raw text of each document
    formatted_results = []  # Stores token-NER results for each document

    current_document = []  # Stores sentences for the current document
    current_formatted = []  # Stores formatted token-NER pairs for the current document

    for index, row in dataset.iterrows():
        word = str(row["word"])  # Convert to string to handle NaN cases
        ner_tag = row["ner"]

        # If we encounter '-DOCSTART-', we recognize it as the start of a new document
        if word == "-DOCSTART-":
            if current_document:  # Save the previous document if it exists
                raw_texts.append(" ".join(current_document))
                formatted_results.append(current_formatted)
            
            # Reset for the new document
            current_document = []
            current_formatted = []
            continue  # Skip adding '-DOCSTART-' to the text

        # If the word is a sentence delimiter (like a period or newline), finalize the sentence
        if word == "." or word == "\n":
            if current_document:  # Avoid adding empty sentences
                current_document.append(word)  # Preserve punctuation
        else:
            # Append token and formatted result
            current_document.append(word)
            current_formatted.append(f"Token: {word}, NER_tag: {ner_tag}")

    # Ensure the last document is added if it wasn't finalized
    if current_document:
        raw_texts.append(" ".join(current_document))
        formatted_results.append(current_formatted)

    return raw_texts, formatted_results


def reformat_and_print():
    # Download the Kaggle dataset
    path = kagglehub.dataset_download("alaakhaled/conll003-englishversion")
    
    # Path to the 'train.txt' file
    train_file = os.path.join(path, 'train.txt')

    # Ensure the file exists before loading
    if not os.path.exists(train_file):
        raise FileNotFoundError("Dataset file 'train.txt' not found.")

    # Load the dataset with space-separated columns (handling multiple spaces between columns)
    dataset = pd.read_csv(train_file, sep=r"\s+", header=None, names=["word", "pos", "chunk", "ner", "other"])

    # Handle NaN in the 'other' column by filling with 'O'
    dataset['other'] = dataset['other'].fillna('O').astype(str)

    # Drop empty rows that may exist in the dataset
    dataset = dataset.dropna(how="all")

    # Generate raw texts and formatted results
    raw_texts, formatted_results = generate_raw_text_and_format(dataset)

    # Print only the first document's data for verification
    print("\n---- First Document (Raw Text) ----\n")
    print(raw_texts[0])  # Print first document

    print("\n---- First Document (Formatted Tokens) ----\n")
    print(formatted_results[0])  # Print first document's formatted results


if __name__ == "__main__":
    reformat_and_print()
