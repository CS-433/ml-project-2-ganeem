from typing import List, Tuple
from pathlib import Path


class TextDataset:
    """
    Dataset class for text data with ID prefixes.
    Format expected: 'ID,text'
    """

    def __init__(self, data_dir: str, filename: str):
        self.data_path = Path(data_dir) / filename
        self.ids, self.texts = self._load_texts()

    def _load_texts(self) -> Tuple[List[str], List[str]]:
        """
        Load texts from file, separating IDs and texts.

        Returns:
            Tuple of (ids, texts)
        """
        ids = []
        texts = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split on first comma only, as text might contain commas
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        id_str, text = parts
                        ids.append(id_str.strip())
                        texts.append(text.strip())
                    else:
                        print(f"Warning: Skipping malformed line: {line}")

        return ids, texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Get item by index.

        Returns:
            Tuple of (id, text)
        """
        return self.ids[idx], self.texts[idx]
