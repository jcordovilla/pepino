"""
Text processing and NLP utilities for Discord message analysis
"""
import re
import spacy
import nltk
from collections import Counter, defaultdict
from typing import List, Dict, Any


# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    nltk.download('punkt')
    nltk.download('stopwords')


# Load spaCy model for advanced NLP
def load_spacy_model():
    """Load spaCy model with error handling"""
    try:
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        return nlp


def preprocess_text(text: str) -> str:
    """Clean and preprocess text for analysis"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'<@!?\d+>', '', text)
    # Remove emojis
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()


def clean_content(text):
    """Advanced text cleaning for topic analysis"""
    # Remove Discord-specific patterns
    text = re.sub(r'<@[!&]?\d+>', '', text)  # Remove mentions
    text = re.sub(r'<#\d+>', '', text)  # Remove channel references
    text = re.sub(r'<:\w+:\d+>', '', text)  # Remove custom emojis
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'`[^`]+`', '', text)  # Remove inline code
    text = re.sub(r'[🎯🏷️💡🔗🔑📝🌎🏭]', '', text)  # Remove emoji noise
    text = re.sub(r'\b(time zone|buddy group|display name|main goal|learning topics)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(the server|the session|the recording|the future)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(messages?|channel|group|topic|session|meeting)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_content_extended(text):
    """Extended text cleaning with additional patterns"""
    text = re.sub(r'<@[!&]?\d+>', '', text)  # Remove mentions
    text = re.sub(r'<#\d+>', '', text)  # Remove channel references
    text = re.sub(r'<:\w+:\d+>', '', text)  # Remove custom emojis
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'`[^`]+`', '', text)  # Remove inline code
    text = re.sub(r'[🎯🏷️💡🔗🔑📝🌎🏭]', '', text)  # Remove emoji noise
    text = re.sub(r'\b(time zone|buddy group|display name|main goal|learning topics)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(the server|the session|the recording|the future)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(messages?|channel|group|topic|session|meeting)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*(minutes?|hours?|days?|weeks?|months?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_complex_concepts(doc):
    """Advanced concept extraction using spaCy"""
    concepts = []
    
    # Extract compound subjects with their predicates
    for token in doc:
        if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
            subject_span = doc[token.left_edge.i:token.right_edge.i+1]
            if token.head.pos_ == "VERB":
                verb = token.head
                predicate_parts = [verb.text]
                for child in verb.children:
                    if child.dep_ in ["dobj", "pobj", "attr", "prep"]:
                        obj_span = doc[child.left_edge.i:child.right_edge.i+1]
                        predicate_parts.append(obj_span.text)
                if len(predicate_parts) > 1:
                    full_concept = f"{subject_span.text} {' '.join(predicate_parts)}"
                    if len(full_concept.split()) >= 3 and len(full_concept) > 15:
                        concepts.append(full_concept.lower().strip())
    
    # Extract extended noun phrases
    for chunk in doc.noun_chunks:
        extended_phrase = chunk.text
        for token in chunk:
            for child in token.children:
                if child.dep_ == "prep":
                    prep_phrase = doc[child.i:child.right_edge.i+1]
                    extended_phrase += f" {prep_phrase.text}"
        if len(extended_phrase.split()) >= 3 and len(extended_phrase) > 20:
            concepts.append(extended_phrase.lower().strip())
    
    # Extract technical compounds
    for i, token in enumerate(doc[:-2]):
        if (token.pos_ in ["NOUN", "PROPN"] and 
            doc[i+1].pos_ in ["NOUN", "PROPN", "ADJ"] and 
            doc[i+2].pos_ in ["NOUN", "PROPN"]):
            compound = f"{token.text} {doc[i+1].text} {doc[i+2].text}"
            j = i + 3
            while j < len(doc) and doc[j].pos_ in ["NOUN", "PROPN"] and j < i + 6:
                compound += f" {doc[j].text}"
                j += 1
            if len(compound.split()) >= 3:
                concepts.append(compound.lower())
    
    return concepts


def get_analysis_patterns():
    """Get predefined patterns for analysis"""
    tech_patterns = [
        r'\b(AI|ML|LLM|GPT|algorithm|model|neural|API|cloud|automation|pipeline|framework|metrics)\b',
        r'\b(python|javascript|typescript|react|node|docker|kubernetes|aws|azure)\b',
        r'\b(database|sql|nosql|analytics|visualization|dashboard|metrics)\b'
    ]
    
    business_patterns = [
        r'\b(team|collaboration|workflow|process|efficiency|optimization|integration|strategy|growth|solution|deployment)\b',
        r'\b(strategy|roadmap|KPI|ROI|revenue|growth|market|customer|client)\b',
        r'\b(product|service|solution|platform|integration|deployment|scale)\b'
    ]
    
    innovation_patterns = [
        r'\b(innovation|transformation|disruption|breakthrough|cutting.edge)\b',
        r'\b(future|trend|emerging|next.gen|state.of.the.art|revolutionary)\b',
        r'\b(experiment|prototype|pilot|proof.of_concept|MVP|beta)\b'
    ]
    
    return tech_patterns, business_patterns, innovation_patterns


def get_topic_analysis_patterns():
    """Get focused patterns for topic analysis"""
    tech_patterns = [
        r'\b(AI|ML|LLM|GPT|algorithm|model|neural|API|cloud|automation|pipeline|framework|metrics)\b'
    ]
    
    business_patterns = [
        r'\b(team|collaboration|workflow|process|efficiency|optimization|integration|strategy|growth|solution|deployment)\b'
    ]
    
    return tech_patterns, business_patterns


# Initialize NLP components when module is imported
download_nltk_data()
nlp = load_spacy_model()
