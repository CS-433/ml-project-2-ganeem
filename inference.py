from typing import List
import hydra
import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from scipy.special import softmax
from tqdm import tqdm

from src.model.sentiment_model import SentimentModel
from src.datasets.text_dataset import TextDataset
from src.utils.text_utils import preprocess_batch
from src.utils.init_utils import set_random_seed


class Inferencer:
    """
    Class for running inference on text data.
    """
    def __init__(
        self,
        model: SentimentModel,
        config: DictConfig,
        device: str,
    ):
        self.model = model
        self.config = config
        self.device = device

    def get_binary_prediction(self, scores: np.ndarray) -> int:
        """
        Convert model scores to binary prediction (1 or -1).
        If the highest score is neutral, use second highest score.
        
        Args:
            scores: Array of sentiment scores (negative, neutral, positive)
            
        Returns:
            1 for positive sentiment, -1 for negative
        """
        sorted_indices = np.argsort(scores)[::-1]
        
        if self.model.config.id2label[sorted_indices[0]] == 'neutral':
            main_score_idx = sorted_indices[1]
        else:
            main_score_idx = sorted_indices[0]
            
        sentiment = self.model.config.id2label[main_score_idx]
        return 1 if sentiment == 'positive' else -1

    def predict_batch(
        self,
        texts: List[str]
    ) -> List[int]:
        """
        Make predictions for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of binary predictions (1 or -1)
        """
        processed_texts = preprocess_batch(texts)
        outputs = self.model.forward(
            processed_texts,
            max_length=self.config.model.max_length
        )
        
        scores = outputs.cpu().numpy()
        predictions = []
        
        for score in scores:
            softmax_scores = softmax(score)
            prediction = self.get_binary_prediction(softmax_scores)
            predictions.append(prediction)
        
        return predictions

    def run_inference(
        self,
        dataset: TextDataset,
    ) -> pd.DataFrame:
        """
        Run inference on the entire dataset.
        
        Args:
            dataset: Text dataset
            
        Returns:
            DataFrame with predictions
        """
        results = []
        ids = []
        batch_size = self.config.inference.batch_size
        assert batch_size > 0, "Batch size must be positive"
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
            batch_items = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            batch_ids = [item[0] for item in batch_items]
            batch_texts = [item[1] for item in batch_items]
            
            batch_predictions = self.predict_batch(batch_texts)
            results.extend(batch_predictions)
            ids.extend(batch_ids)
            
        return pd.DataFrame({
            'Id': ids,
            'Prediction': results
        })

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config: DictConfig) -> None:
    """
    Main inference script.
    """
    set_random_seed(config.inference.seed)

    # set device
    if config.inference.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inference.device
    print(f"Using device: {device}")

    model = SentimentModel(
        model_name=config.model.name,
        device=device
    )

    dataset = TextDataset(
        data_dir=config.data.data_dir,
        filename=config.data.input_file
    )
    print(f"Loaded {len(dataset)} texts")

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
    )

    predictions_df = inferencer.run_inference(dataset)

    output_path = Path(config.data.output_dir) / config.data.output_file
    output_path.parent.mkdir(exist_ok=True, parents=True)
    predictions_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print("\nPrediction statistics:")
    print(predictions_df['Prediction'].value_counts())


if __name__ == "__main__":
    main()
