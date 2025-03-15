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

# Function to generate text with cache augmentation
def generate_with_cache(query, max_length=1024, max_new_tokens=150):
    relevant_texts = retrieve_from_cache(query)
    context = "Here is relevant context: " + " ".join(relevant_texts) + "\nUser Query: " + query
    inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3, temperature=0.6, top_p=0.95, top_k=50, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    correlation_percentage = calculate_correlation(generated_text, relevant_texts)
    return generated_text, correlation_percentage

# Function to generate text with knowledge augmentation
def generate_with_knowledge(query, max_length=1024, max_new_tokens=150):
    knowledge_text = retrieve_knowledge(query)
    context = "Here is relevant knowledge: " + knowledge_text + "\nUser Query: " + query
    inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3, temperature=0.6, top_p=0.95, top_k=50, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    correlation_percentage = calculate_correlation(generated_text, [knowledge_text])
    return generated_text, correlation_percentage

# Function to generate text using GraphRAG
def generate_with_graph_rag(query):
    G = nx.Graph()
    knowledge_text = retrieve_knowledge(query)
    relevant_texts = retrieve_from_cache(query)
    all_texts = [query] + relevant_texts + [knowledge_text]
    for i, text in enumerate(all_texts):
        G.add_node(i, text=text)
        for j in range(i):
            similarity = SequenceMatcher(None, text, all_texts[j]).ratio()
            G.add_edge(i, j, weight=similarity)
    
    central_nodes = sorted(G.nodes, key=lambda x: nx.degree(G, x), reverse=True)[:3]
    context = " ".join([G.nodes[n]['text'] for n in central_nodes])
    context += "\nUser Query: " + query
    
    inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=150, no_repeat_ngram_size=3, temperature=0.6, top_p=0.95, top_k=50, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    correlation_percentage = calculate_correlation(generated_text, all_texts)
    return generated_text, correlation_percentage, G

# Streamlit UI
st.title("CAG vs KAG vs GraphRAG AI Text Generator")
query = st.text_area("Enter your query:")
generate_button = st.button("Generate Response")

if generate_button:
    if query:
        add_to_cache(query)
        cag_response, cag_correlation = generate_with_cache(query)
        kag_response, kag_correlation = generate_with_knowledge(query)
        graph_response, graph_correlation, graph = generate_with_graph_rag(query)
        
        st.subheader("Cache-Augmented Response (CAG):")
        st.write(cag_response)
        st.write(f"Correlation with Cache: {cag_correlation}%")
        
        st.subheader("Knowledge-Augmented Response (KAG):")
        st.write(kag_response)
        st.write(f"Correlation with External Knowledge: {kag_correlation}%")
        
        st.subheader("Graph-RAG Response:")
        st.write(graph_response)
        st.write(f"Correlation with Knowledge Graph: {graph_correlation}%")
        
        # Generate visualization
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
        st.pyplot(plt)
        
        # Correlation comparison chart
        data = pd.DataFrame({"Method": ["CAG", "KAG", "GraphRAG"], "Correlation (%)": [cag_correlation, kag_correlation, graph_correlation]})
        fig, ax = plt.subplots()
        sns.barplot(x="Method", y="Correlation (%)", data=data, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please enter a query.")
        
# Display cache content
if st.checkbox("Show Cache Content"):
    st.subheader("Current Cache")
    for idx, entry in enumerate(cache):
        st.write(f"{idx+1}. {entry}")
