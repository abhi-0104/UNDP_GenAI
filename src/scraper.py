import requests
from bs4 import BeautifulSoup
# URL of the webpage listing FFD4 documents
url = "https://financing.desa.un.org/ffd4/elementspaperinputs"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links that point to PDFs
pdf_links = []
for link in soup.find_all("a", href=True):
    if link["href"].endswith(".pdf"):
        full_url = link["href"]
        pdf_links.append(full_url)

print(f"Found {len(pdf_links)} PDFs")
