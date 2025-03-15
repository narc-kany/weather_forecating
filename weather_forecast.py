import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
from difflib import SequenceMatcher

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

# Function to calculate similarity between cache and generated response
def calculate_correlation(generated_text, cache):
    combined_cache_text = " ".join(cache)
    similarity = SequenceMatcher(None, generated_text, combined_cache_text).ratio()
    return round(similarity * 100, 2)

# Function to generate text with cache augmentation
def generate_with_cache(query, max_length=1024, max_new_tokens=150):
    # Retrieve relevant cached information
    relevant_texts = retrieve_from_cache(query)
    
    # Combine retrieved cache with query
    context = "Here is relevant context: " + " ".join(relevant_texts) + "\nUser Query: " + query
    
    # Tokenize input
    inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,  # Reduce repetition further
            temperature=0.6,  # More controlled randomness
            top_p=0.95,  # Increase sampling diversity
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate correlation percentage
    correlation_percentage = calculate_correlation(generated_text, relevant_texts)
    
    return generated_text, correlation_percentage

# Streamlit UI
st.title("Cache-Augmented AI Text Generator")
st.write("This AI generates responses based on both new queries and cached knowledge.")

query = st.text_area("Enter your query:")
generate_button = st.button("Generate Response")

if generate_button:
    if query:
        add_to_cache(query)
        response, correlation = generate_with_cache(query)
        st.subheader("Generated Response:")
        st.write(response)
        st.subheader("Correlation with Cache:")
        st.write(f"{correlation}% similar to cached knowledge.")
    else:
        st.warning("Please enter a query.")

# Display cache content
if st.checkbox("Show Cache Content"):
    st.subheader("Current Cache")
    for idx, entry in enumerate(cache):
        st.write(f"{idx+1}. {entry}")
