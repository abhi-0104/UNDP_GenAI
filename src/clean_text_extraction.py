import os
import requests
import pdfplumber
from io import BytesIO
from collections import Counter
from scraper import pdf_links

output_folder = "extracted_texts"
os.makedirs(output_folder, exist_ok=True)

def extract_cleaned_text(pdf_file):
    full_text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Get words with font information
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            # Get most common font size (assumed to be main text)
            font_sizes = [round(word['size'], 1) for word in words]
            if not font_sizes:
                continue
            common_size = Counter(font_sizes).most_common(1)[0][0]

            # Filter out footnote-style text (small font or low on page)
            filtered_words = [
                word for word in words
                if round(word['size'], 1) >= common_size and word['top'] < page.height * 0.85
            ]

            page_text = " ".join(word['text'] for word in filtered_words)
            if page_text:
                full_text += page_text.strip() + "\n\n"

    return full_text.strip()

# Process PDF links
for idx, url in enumerate(pdf_links, start=1):
    try:
        print(f"Processing PDF {idx}: {url}")
        response = requests.get(url)
        response.raise_for_status()

        pdf_file = BytesIO(response.content)
        cleaned_text = extract_cleaned_text(pdf_file)

        output_path = os.path.join(output_folder, f"doc_{idx}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"Saved to {output_path}\n")

    except Exception as e:
        print(f"Failed to process {url}: {e}\n")
