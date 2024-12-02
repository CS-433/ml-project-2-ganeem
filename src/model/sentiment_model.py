from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch


class SentimentModel:
    """
    Wrapper class for sentiment analysis model.
    """

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model.eval()

    def forward(self, texts: list, max_length: int) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length

        Returns:
            Model output tensor
        """
        encoded_input = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded_input)

        return output[0]
