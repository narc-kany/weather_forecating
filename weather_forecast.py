import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
from difflib import SequenceMatcher
import wikipedia
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and tokenizer
def load_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Cache setup
cache_size = 5  # Maximum number of cache entries
cache = deque(maxlen=cache_size)

# Predefined cache entries about weather forecasting
predefined_cache = [
    "Weather forecasting is the application of science and technology to predict atmospheric conditions.",
    "Meteorologists use satellite imagery, radar, and computer models to predict the weather.",
    "Modern weather forecasts rely on AI and machine learning for more accurate predictions.",
    "Severe weather alerts help warn people about hurricanes, tornadoes, and storms in advance.",
    "Climate change affects long-term weather patterns, making forecasting even more critical."
]

# Load predefined cache into the deque
for entry in predefined_cache:
    cache.append(entry)

# Function to add content to cache
def add_to_cache(text):
    cache.append(text)

# Function to retrieve relevant content from cache
def retrieve_from_cache(query, k=3):
    return list(cache)[-k:]

# Function to retrieve knowledge from Wikipedia
def retrieve_knowledge(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple topics found: " + ", ".join(e.options[:3])
    except wikipedia.exceptions.PageError:
        return "No relevant Wikipedia page found."

# Function to calculate similarity
def calculate_correlation(generated_text, reference_texts):
    combined_reference_text = " ".join(reference_texts)
    similarity = SequenceMatcher(None, generated_text, combined_reference_text).ratio()
    return round(similarity * 100, 2)

# Additional Evaluation Metrics
def calculate_bleu(generated_text, reference_texts):
    reference = [ref.split() for ref in reference_texts]
    candidate = generated_text.split()
    return sentence_bleu(reference, candidate) * 100

def calculate_rouge(generated_text, reference_texts):
    rouge = Rouge()
    scores = rouge.get_scores(generated_text, " ".join(reference_texts))
    return scores[0]['rouge-l']['f'] * 100

def calculate_cosine_similarity(generated_text, reference_texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([generated_text] + reference_texts)
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:])
    return cosine_sim.mean() * 100

# Function to generate text with cache augmentation
def generate_with_cache(query, max_length=1024, max_new_tokens=150):
    relevant_texts = retrieve_from_cache(query)
    context = "Here is relevant context: " + " ".join(relevant_texts) + "\nUser Query: " + query
    inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3, temperature=0.6, top_p=0.95, top_k=50, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, calculate_correlation(generated_text, relevant_texts), calculate_bleu(generated_text, relevant_texts), calculate_rouge(generated_text, relevant_texts), calculate_cosine_similarity(generated_text, relevant_texts)

# Streamlit UI
st.title("CAG vs KAG vs GraphRAG AI Text Generator")
query = st.text_area("Enter your query:")
generate_button = st.button("Generate Response")

if generate_button:
    if query:
        add_to_cache(query)
        cag_response, cag_correlation, cag_bleu, cag_rouge, cag_cosine = generate_with_cache(query)
        
        # Visualization
        data = pd.DataFrame({
            "Metric": ["Correlation", "BLEU", "ROUGE-L", "Cosine Similarity"],
            "CAG": [cag_correlation, cag_bleu, cag_rouge, cag_cosine]
        })
        
        fig, ax = plt.subplots()
        sns.barplot(x="Metric", y="CAG", data=data, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please enter a query.")
# Display cache content
if st.checkbox("Show Cache Content"):
    st.subheader("Current Cache")
    for idx, entry in enumerate(cache):
        st.write(f"{idx+1}. {entry}")
