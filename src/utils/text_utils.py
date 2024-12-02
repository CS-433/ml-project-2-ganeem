from typing import List

def preprocess_text(text: str) -> str:
    """Preprocess single text."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def preprocess_batch(texts: List[str]) -> List[str]:
    """Preprocess a batch of texts."""
    return [preprocess_text(text) for text in texts]