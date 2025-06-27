"""
Модель BERT для анализа настроений
Используется предобученная модель BERT для классификации настроений
"""

import logging
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import joblib
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SentimentBERTModel:
    """Модель BERT для анализа рыночных настроений"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None

        # Параметры модели
        self.model_name = config.get("model_name", "nlptown/bert-base-multilingual-uncased-sentiment")
        self.max_length = config.get("max_length", 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Загрузка предобученной модели BERT"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name)
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)
            logger.info(f"Модель BERT успешно загружена: {self.model_name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели BERT: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Анализ настроений для списка текстов

        Args:
            texts: Список текстов для анализа

        Returns:
            Список словарей с результатами анализа
        """
        try:
            if not self.sentiment_pipeline:
                raise ValueError("Модель BERT не загружена. Используйте метод load_model().")

            predictions = self.sentiment_pipeline(texts, truncation=True, max_length=self.max_length)
            results = []

            for text, prediction in zip(texts, predictions):
                label = prediction["label"]
                score = prediction["score"]

                # Преобразование меток в числовую шкалу
                sentiment_score = self._convert_label_to_score(label)
                results.append({
                    "text": text,
                    "sentiment_score": sentiment_score,
                    "confidence": score,
                    "label": label
                })

            return results
        except Exception as e:
            logger.error(f"Ошибка при анализе настроений: {e}")
            return []

    def _convert_label_to_score(self, label: str) -> float:
        """Преобразование меток BERT в числовую шкалу (-1 до 1)"""
        try:
            label_mapping = {
                "1 star": -1.0,
                "2 stars": -0.5,
                "3 stars": 0.0,
                "4 stars": 0.5,
                "5 stars": 1.0
            }
            return label_mapping.get(label.lower(), 0.0)
        except Exception as e:
            logger.error(f"Ошибка преобразования метки: {e}")
            return 0.0

    def save(self, filepath: str) -> None:
        """Сохранение конфигурации и состояния модели"""
        try:
            joblib.dump({
                "config": self.config,
                "model_name": self.model_name
            }, filepath)
            logger.info(f"Конфигурация модели BERT сохранена в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели BERT: {e}")

    def load(self, filepath: str) -> None:
        """Загрузка конфигурации модели"""
        try:
            data = joblib.load(filepath)
            self.config = data["config"]
            self.model_name = data["model_name"]
            self.load_model()
            logger.info(f"Конфигурация модели BERT загружена из {filepath}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели BERT: {e}")