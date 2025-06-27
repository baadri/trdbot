"""
Ансамблевая модель голосования
Объединяет результаты нескольких моделей для генерации торгового сигнала
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ансамблевая модель голосования для объединения результатов моделей"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_weights = config.get("ensemble_weights", {})
        self.confidence_threshold = config.get("confidence_threshold", 0.75)

    def predict(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Генерация финального сигнала на основе голосования моделей

        Args:
            predictions: Список прогнозов от отдельных моделей

        Returns:
            Финальный прогноз с учетом весов и уверенности
        """
        try:
            if not predictions:
                raise ValueError("Список прогнозов пуст")

            # Подсчет голосов
            votes = {"buy": 0, "sell": 0, "hold": 0}
            total_confidence = 0.0
            weighted_votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

            for prediction in predictions:
                model_type = prediction["model_type"]
                signal = prediction["prediction"]
                confidence = prediction["confidence"]
                weight = self.model_weights.get(model_type, 1.0)

                if signal in votes:
                    votes[signal] += 1
                    weighted_votes[signal] += confidence * weight
                    total_confidence += confidence * weight

            # Нормализация взвешенных голосов
            for signal in weighted_votes:
                if total_confidence > 0:
                    weighted_votes[signal] /= total_confidence

            # Выбор финального сигнала
            final_signal = max(weighted_votes, key=weighted_votes.get)
            final_confidence = weighted_votes[final_signal]

            # Проверка порога уверенности
            if final_confidence < self.confidence_threshold:
                logger.warning("Недостаточно уверенности для принятия решения")
                return {"signal": "hold", "confidence": final_confidence}

            return {
                "signal": final_signal,
                "confidence": final_confidence,
                "votes": votes,
                "weighted_votes": weighted_votes,
                "model_contributions": predictions
            }
        except Exception as e:
            logger.error(f"Ошибка в ансамблевом прогнозе: {e}")
            return {"signal": "hold", "confidence": 0.0}

    def get_model_weights(self) -> Dict[str, float]:
        """Возвращает текущие веса моделей"""
        return self.model_weights