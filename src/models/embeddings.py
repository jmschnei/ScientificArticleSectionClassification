from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Sentences we want sentence embeddings for
sentences = ["The quick brown fox jumps over the lazy dog.",
             "Now is the time for all good men to come to the aid of their country."]

class HuggingFaceEmbeddingModel(Embeddings):
    def __init__(self):
        #self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', device='cpu', trust_remote_code=True)
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    def embed_query(self, text):
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
