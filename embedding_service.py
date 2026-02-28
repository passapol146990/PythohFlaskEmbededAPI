from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded successfully!")

    def generate_embedding(self, text: str) -> List[float]:
        if not text or not isinstance(text, str):
            return []

        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        embedding = self.model.encode(cleaned_text, convert_to_tensor=False).tolist()
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts if text and isinstance(text, str)]

        if not cleaned_texts:
            return []

        embeddings = self.model.encode(cleaned_texts, convert_to_tensor=False).tolist()
        return embeddings

emdeb = None
def useEmdeb():
    global emdeb
    if emdeb is None:
        emdeb = EmbeddingService()
    return emdeb


