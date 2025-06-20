import os
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
from itertools import islice

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Folder paths
input_folder = "extracted_texts"
output_folder = "eda_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load all documents
documents = []
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({"filename": filename, "text": text})

print(f"✅ Loaded {len(documents)} documents")

# Token collection and NER
all_tokens = []
all_gpe = []

for doc in documents:
    spacy_doc = nlp(doc["text"])

    # Tokens
    tokens = [token.text.lower() for token in spacy_doc if token.is_alpha and not token.is_stop]
    all_tokens.extend(tokens)

    # Country mentions (GPE)
    gpes = [ent.text for ent in spacy_doc.ents if ent.label_ == "GPE"]
    all_gpe.extend(gpes)

# Word frequency
word_freq = Counter(all_tokens)
top_words = word_freq.most_common(20)
words, freqs = zip(*top_words)

# Plot top words
plt.figure(figsize=(10, 5))
plt.bar(words, freqs, color='green')
plt.xticks(rotation=45)
plt.title("Top 20 Words (spaCy)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_words_spacy.png"))
plt.close()

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "wordcloud_spacy.png"))
plt.close()

# Country frequency
country_freq = Counter(all_gpe)
country_df = pd.DataFrame(country_freq.most_common(20), columns=["Country", "Frequency"])
country_df.to_csv(os.path.join(output_folder, "top_country_mentions.csv"), index=False)

# Sentiment analysis
sentiments = []
for doc in documents:
    blob = TextBlob(doc["text"])
    polarity = blob.sentiment.polarity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    sentiments.append({"filename": doc["filename"], "polarity": polarity, "sentiment": sentiment})

sentiment_df = pd.DataFrame(sentiments)
sentiment_df.to_csv(os.path.join(output_folder, "document_sentiments.csv"), index=False)

# Bigrams
def generate_bigrams(tokens):
    return zip(tokens, tokens[1:])

bigrams = generate_bigrams(all_tokens)
bigram_freq = Counter(bigrams)
top_bigrams = list(islice(bigram_freq.items(), 20))
bigram_df = pd.DataFrame([" ".join(bg) for bg, _ in top_bigrams], columns=["Bigram"])
bigram_df["Frequency"] = [freq for _, freq in top_bigrams]
bigram_df.to_csv(os.path.join(output_folder, "top_bigrams.csv"), index=False)

print("🎉 EDA completed — outputs saved in:", output_folder)
