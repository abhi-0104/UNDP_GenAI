import os
import re
import spacy
import pycountry
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
from itertools import islice

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input/output
input_folder = "extracted_texts"
output_folder = "eda_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load documents
documents = []
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({"filename": filename, "text": text})

print(f"✅ Loaded {len(documents)} documents")

# Initialize containers
all_tokens = []
all_gpe = []
climate_sentences = []
climate_sentiments = []
climate_policy_mentions = []

for doc in documents:
    spacy_doc = nlp(doc["text"])

    # Tokens
    tokens = [token.text.lower() for token in spacy_doc if token.is_alpha and not token.is_stop]
    all_tokens.extend(tokens)

    # GPEs (countries/regions)
    all_gpe.extend([ent.text.strip() for ent in spacy_doc.ents if ent.label_ == "GPE"])

    # Climate-related sentences
    for sent in spacy_doc.sents:
        sentence_text = sent.text.lower()
        if "climate" in sentence_text or "green finance" in sentence_text or "climate finance" in sentence_text:
            climate_sentences.append(sent.text)

            # Climate policy sentence?
            if any(policy_kw in sentence_text for policy_kw in ["policy", "mechanism", "strategy", "framework", "instrument"]):
                climate_policy_mentions.append(sent.text)

            # Sentiment
            blob = TextBlob(sent.text)
            polarity = blob.sentiment.polarity
            sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            climate_sentiments.append({"sentence": sent.text.strip(), "sentiment": sentiment, "polarity": polarity})

# === Filter Valid Countries ===
valid_countries = {country.name for country in pycountry.countries}
country_aliases = {
    "US": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
}

normalized_countries = []
for gpe in all_gpe:
    if gpe in valid_countries:
        normalized_countries.append(gpe)
    elif gpe in country_aliases:
        normalized_countries.append(country_aliases[gpe])

# === Save Country Mentions ===
country_freq = Counter(normalized_countries)
country_df = pd.DataFrame(country_freq.most_common(30), columns=["Country", "Mentions"])
country_df.to_csv(os.path.join(output_folder, "country_mentions.csv"), index=False)

# === Sentiment CSV ===
sentiment_df = pd.DataFrame(climate_sentiments)
sentiment_df.to_csv(os.path.join(output_folder, "climate_sentiments.csv"), index=False)

# === Climate Policy Mentions ===
with open(os.path.join(output_folder, "climate_policy_mentions.txt"), "w", encoding="utf-8") as f:
    for sent in climate_policy_mentions:
        f.write(sent.strip() + "\n")

# === Word Frequency Bar Plot ===
word_freq = Counter(all_tokens)
top_words = word_freq.most_common(25)
words, freqs = zip(*top_words)

plt.figure(figsize=(10, 5))
plt.bar(words, freqs, color='green')
plt.xticks(rotation=45)
plt.title("Top 25 Frequent Words")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_words.png"))
plt.close()

# === Word Cloud ===
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate_from_frequencies(word_freq)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "wordcloud.png"))
plt.close()

# === Bigram Frequency ===
def generate_bigrams(tokens):
    return zip(tokens, tokens[1:])

bigram_freq = Counter(generate_bigrams(all_tokens))
top_bigrams = list(islice(bigram_freq.items(), 25))
bigram_df = pd.DataFrame([" ".join(bg) for bg, _ in top_bigrams], columns=["Bigram"])
bigram_df["Frequency"] = [freq for _, freq in top_bigrams]
bigram_df.to_csv(os.path.join(output_folder, "top_bigrams.csv"), index=False)

# === Theme Extraction from Noun Chunks ===
chunk_counter = Counter()
for doc in documents:
    doc_chunks = nlp(doc["text"])
    chunk_counter.update([chunk.text.lower() for chunk in doc_chunks.noun_chunks if len(chunk.text.split()) > 1])

theme_df = pd.DataFrame(chunk_counter.most_common(30), columns=["Noun Phrase", "Frequency"])
theme_df.to_csv(os.path.join(output_folder, "key_themes.csv"), index=False)

print("🎯 All EDA outputs saved to:", output_folder)
