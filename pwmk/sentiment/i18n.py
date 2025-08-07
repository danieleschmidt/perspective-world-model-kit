"""
Internationalization (i18n) support for sentiment analysis.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import locale
import logging
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for sentiment analysis."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"


@dataclass
class LanguageConfig:
    """Configuration for a specific language."""
    code: str
    name: str
    model_name: Optional[str] = None
    tokenizer_config: Optional[Dict[str, Any]] = None
    sentiment_labels: Optional[Dict[str, str]] = None
    preprocessing_rules: Optional[Dict[str, Any]] = None


class MultilingualSentimentAnalyzer:
    """
    Sentiment analyzer with multi-language support.
    """
    
    def __init__(
        self,
        default_language: str = "en",
        supported_languages: Optional[List[str]] = None
    ):
        self.default_language = default_language
        self.supported_languages = supported_languages or [
            "en", "es", "fr", "de", "ja", "zh"
        ]
        
        # Language configurations
        self.language_configs: Dict[str, LanguageConfig] = {}
        self._load_language_configurations()
        
        # Language-specific models
        self.language_models: Dict[str, Any] = {}
        self._initialize_language_models()
        
        # Translation cache for cross-language analysis
        self.translation_cache: Dict[Tuple[str, str, str], str] = {}
        
        logger.info(f"Multilingual analyzer initialized with languages: {self.supported_languages}")
        
    def _load_language_configurations(self) -> None:
        """Load language-specific configurations."""
        # Default configurations for each language
        default_configs = {
            "en": LanguageConfig(
                code="en",
                name="English",
                model_name="bert-base-uncased",
                sentiment_labels={
                    "negative": "negative",
                    "neutral": "neutral", 
                    "positive": "positive"
                }
            ),
            "es": LanguageConfig(
                code="es",
                name="Español",
                model_name="dccuchile/bert-base-spanish-wwm-uncased",
                sentiment_labels={
                    "negative": "negativo",
                    "neutral": "neutral",
                    "positive": "positivo"
                }
            ),
            "fr": LanguageConfig(
                code="fr",
                name="Français",
                model_name="camembert-base",
                sentiment_labels={
                    "negative": "négatif",
                    "neutral": "neutre",
                    "positive": "positif"
                }
            ),
            "de": LanguageConfig(
                code="de",
                name="Deutsch",
                model_name="bert-base-german-cased",
                sentiment_labels={
                    "negative": "negativ",
                    "neutral": "neutral",
                    "positive": "positiv"
                }
            ),
            "ja": LanguageConfig(
                code="ja",
                name="日本語",
                model_name="cl-tohoku/bert-base-japanese",
                sentiment_labels={
                    "negative": "ネガティブ",
                    "neutral": "中立",
                    "positive": "ポジティブ"
                }
            ),
            "zh": LanguageConfig(
                code="zh",
                name="中文",
                model_name="bert-base-chinese",
                sentiment_labels={
                    "negative": "负面",
                    "neutral": "中性",
                    "positive": "正面"
                }
            )
        }
        
        for lang_code in self.supported_languages:
            if lang_code in default_configs:
                self.language_configs[lang_code] = default_configs[lang_code]
            else:
                # Fallback configuration
                self.language_configs[lang_code] = LanguageConfig(
                    code=lang_code,
                    name=lang_code.upper(),
                    model_name="bert-base-multilingual-cased",
                    sentiment_labels={
                        "negative": "negative",
                        "neutral": "neutral",
                        "positive": "positive"
                    }
                )
                
    def _initialize_language_models(self) -> None:
        """Initialize language-specific sentiment models."""
        from .sentiment_analyzer import SentimentAnalyzer
        
        for lang_code, config in self.language_configs.items():
            try:
                if config.model_name:
                    model = SentimentAnalyzer(model_name=config.model_name)
                    self.language_models[lang_code] = model
                    logger.info(f"Loaded model for {config.name} ({lang_code})")
                else:
                    # Use default multilingual model
                    model = SentimentAnalyzer(model_name="bert-base-multilingual-cased")
                    self.language_models[lang_code] = model
                    logger.info(f"Using multilingual model for {lang_code}")
                    
            except Exception as e:
                logger.warning(f"Failed to load model for {lang_code}: {e}")
                # Fallback to English model
                if "en" in self.language_models:
                    self.language_models[lang_code] = self.language_models["en"]
                    logger.info(f"Using English model as fallback for {lang_code}")
                    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO 639-1 language code
        """
        try:
            # Simple heuristic-based language detection
            # In production, you might use langdetect or similar library
            
            # Character-based detection for some languages
            if any('\u4e00' <= char <= '\u9fff' for char in text):  # Chinese characters
                return "zh"
            elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):  # Japanese
                return "ja"
            elif any('\u0600' <= char <= '\u06ff' for char in text):  # Arabic
                return "ar"
            elif any('\u0400' <= char <= '\u04ff' for char in text):  # Cyrillic (Russian)
                return "ru"
                
            # Simple word-based detection for European languages
            text_lower = text.lower()
            
            # Spanish indicators
            if any(word in text_lower for word in ['el', 'la', 'es', 'en', 'de', 'que', 'y', 'a', 'un', 'se']):
                spanish_score = sum(1 for word in ['el', 'la', 'es', 'que', 'y', 'un', 'se'] if word in text_lower)
                if spanish_score >= 2:
                    return "es"
                    
            # French indicators
            if any(word in text_lower for word in ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']):
                french_score = sum(1 for word in ['le', 'de', 'et', 'à', 'être'] if word in text_lower)
                if french_score >= 2:
                    return "fr"
                    
            # German indicators
            if any(word in text_lower for word in ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich']):
                german_score = sum(1 for word in ['der', 'die', 'und', 'den', 'von', 'das'] if word in text_lower)
                if german_score >= 2:
                    return "de"
                    
            # Default to English
            return self.default_language
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return self.default_language
            
    def analyze_text(
        self,
        text: str,
        language: Optional[str] = None,
        return_language_info: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text in specified or detected language.
        
        Args:
            text: Text to analyze
            language: Language code (if None, will be auto-detected)
            return_language_info: Whether to include language detection info
            
        Returns:
            Sentiment analysis results with optional language info
        """
        # Auto-detect language if not specified
        if language is None:
            detected_language = self.detect_language(text)
            logger.debug(f"Detected language: {detected_language}")
        else:
            detected_language = language
            
        # Validate language support
        if detected_language not in self.supported_languages:
            logger.warning(f"Language {detected_language} not supported, using default")
            detected_language = self.default_language
            
        # Get language-specific model
        model = self.language_models.get(detected_language)
        if model is None:
            logger.warning(f"No model for {detected_language}, using default")
            model = self.language_models.get(self.default_language)
            
        if model is None:
            raise ValueError("No sentiment model available")
            
        # Perform sentiment analysis
        try:
            sentiment_scores = model.analyze_text(text)
            
            # Translate sentiment labels if needed
            config = self.language_configs[detected_language]
            if config.sentiment_labels:
                localized_labels = {}
                for eng_label, score in sentiment_scores.items():
                    local_label = config.sentiment_labels.get(eng_label, eng_label)
                    localized_labels[local_label] = score
                sentiment_scores = localized_labels
                
            result = {
                "sentiment": sentiment_scores,
                "language": detected_language
            }
            
            if return_language_info:
                result["language_info"] = {
                    "detected_language": detected_language,
                    "language_name": config.name,
                    "model_used": config.model_name,
                    "confidence": 0.8  # Placeholder confidence score
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for language {detected_language}: {e}")
            
            # Return default sentiment
            return {
                "sentiment": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                "language": detected_language,
                "error": str(e)
            }
            
    def analyze_cross_language(
        self,
        texts: List[Tuple[str, str]],  # (text, language) pairs
        normalize_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment across multiple languages and optionally normalize results.
        
        Args:
            texts: List of (text, language) tuples
            normalize_results: Whether to normalize sentiment labels to English
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for text, language in texts:
            result = self.analyze_text(text, language=language, return_language_info=True)
            
            if normalize_results:
                # Convert localized labels back to English
                sentiment_scores = result["sentiment"]
                config = self.language_configs.get(language, self.language_configs[self.default_language])
                
                if config.sentiment_labels:
                    # Create reverse mapping
                    label_mapping = {v: k for k, v in config.sentiment_labels.items()}
                    
                    normalized_scores = {}
                    for local_label, score in sentiment_scores.items():
                        eng_label = label_mapping.get(local_label, local_label)
                        normalized_scores[eng_label] = score
                        
                    result["sentiment"] = normalized_scores
                    result["normalized"] = True
                    
            results.append(result)
            
        return results
        
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return {
            code: config.name 
            for code, config in self.language_configs.items()
        }
        
    def set_language_preference(self, agent_id: int, language: str) -> None:
        """Set language preference for specific agent."""
        if not hasattr(self, 'agent_languages'):
            self.agent_languages = {}
            
        if language in self.supported_languages:
            self.agent_languages[agent_id] = language
            logger.info(f"Set language preference for agent {agent_id}: {language}")
        else:
            logger.warning(f"Language {language} not supported for agent {agent_id}")
            
    def get_agent_language(self, agent_id: int) -> str:
        """Get language preference for specific agent."""
        if hasattr(self, 'agent_languages'):
            return self.agent_languages.get(agent_id, self.default_language)
        return self.default_language


class LocalizationManager:
    """
    Manager for UI strings and message localization.
    """
    
    def __init__(self, locales_path: Optional[str] = None):
        self.locales_path = locales_path or self._get_default_locales_path()
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_locale = "en"
        
        self._load_translations()
        
    def _get_default_locales_path(self) -> str:
        """Get default path for locale files."""
        current_dir = Path(__file__).parent
        return str(current_dir / "locales")
        
    def _load_translations(self) -> None:
        """Load translation files for all supported languages."""
        translations_data = {
            "en": {
                "sentiment.analysis.started": "Sentiment analysis started",
                "sentiment.analysis.completed": "Sentiment analysis completed",
                "sentiment.analysis.failed": "Sentiment analysis failed",
                "sentiment.positive": "Positive",
                "sentiment.negative": "Negative", 
                "sentiment.neutral": "Neutral",
                "agent.communication.recorded": "Agent communication recorded",
                "belief.updated": "Belief updated",
                "error.invalid_input": "Invalid input provided",
                "error.model_not_found": "Sentiment model not found",
                "error.processing_failed": "Processing failed",
                "status.healthy": "System is healthy",
                "status.degraded": "System performance is degraded",
                "status.unhealthy": "System is unhealthy"
            },
            "es": {
                "sentiment.analysis.started": "Análisis de sentimientos iniciado",
                "sentiment.analysis.completed": "Análisis de sentimientos completado", 
                "sentiment.analysis.failed": "Análisis de sentimientos falló",
                "sentiment.positive": "Positivo",
                "sentiment.negative": "Negativo",
                "sentiment.neutral": "Neutral",
                "agent.communication.recorded": "Comunicación del agente registrada",
                "belief.updated": "Creencia actualizada",
                "error.invalid_input": "Entrada inválida proporcionada",
                "error.model_not_found": "Modelo de sentimientos no encontrado",
                "error.processing_failed": "Procesamiento falló",
                "status.healthy": "El sistema está saludable",
                "status.degraded": "El rendimiento del sistema está degradado",
                "status.unhealthy": "El sistema no está saludable"
            },
            "fr": {
                "sentiment.analysis.started": "Analyse des sentiments démarrée",
                "sentiment.analysis.completed": "Analyse des sentiments terminée",
                "sentiment.analysis.failed": "L'analyse des sentiments a échoué",
                "sentiment.positive": "Positif",
                "sentiment.negative": "Négatif",
                "sentiment.neutral": "Neutre",
                "agent.communication.recorded": "Communication de l'agent enregistrée",
                "belief.updated": "Croyance mise à jour",
                "error.invalid_input": "Entrée invalide fournie",
                "error.model_not_found": "Modèle de sentiment non trouvé",
                "error.processing_failed": "Le traitement a échoué",
                "status.healthy": "Le système est en bonne santé",
                "status.degraded": "Les performances du système sont dégradées",
                "status.unhealthy": "Le système n'est pas en bonne santé"
            },
            "de": {
                "sentiment.analysis.started": "Sentimentanalyse gestartet",
                "sentiment.analysis.completed": "Sentimentanalyse abgeschlossen",
                "sentiment.analysis.failed": "Sentimentanalyse fehlgeschlagen",
                "sentiment.positive": "Positiv",
                "sentiment.negative": "Negativ", 
                "sentiment.neutral": "Neutral",
                "agent.communication.recorded": "Agent-Kommunikation aufgezeichnet",
                "belief.updated": "Überzeugung aktualisiert",
                "error.invalid_input": "Ungültige Eingabe bereitgestellt",
                "error.model_not_found": "Sentiment-Modell nicht gefunden",
                "error.processing_failed": "Verarbeitung fehlgeschlagen",
                "status.healthy": "System ist gesund",
                "status.degraded": "Systemleistung ist beeinträchtigt",
                "status.unhealthy": "System ist ungesund"
            },
            "ja": {
                "sentiment.analysis.started": "感情分析を開始しました",
                "sentiment.analysis.completed": "感情分析が完了しました",
                "sentiment.analysis.failed": "感情分析に失敗しました",
                "sentiment.positive": "ポジティブ",
                "sentiment.negative": "ネガティブ",
                "sentiment.neutral": "中立",
                "agent.communication.recorded": "エージェントの通信が記録されました",
                "belief.updated": "信念が更新されました",
                "error.invalid_input": "無効な入力が提供されました",
                "error.model_not_found": "感情モデルが見つかりません",
                "error.processing_failed": "処理に失敗しました",
                "status.healthy": "システムは健全です",
                "status.degraded": "システムのパフォーマンスが低下しています",
                "status.unhealthy": "システムは不健全です"
            },
            "zh": {
                "sentiment.analysis.started": "情感分析已开始",
                "sentiment.analysis.completed": "情感分析已完成",
                "sentiment.analysis.failed": "情感分析失败",
                "sentiment.positive": "正面",
                "sentiment.negative": "负面",
                "sentiment.neutral": "中性",
                "agent.communication.recorded": "智能体通信已记录",
                "belief.updated": "信念已更新",
                "error.invalid_input": "提供的输入无效",
                "error.model_not_found": "未找到情感模型",
                "error.processing_failed": "处理失败",
                "status.healthy": "系统健康",
                "status.degraded": "系统性能下降",
                "status.unhealthy": "系统不健康"
            }
        }
        
        self.translations = translations_data
        logger.info(f"Loaded translations for {len(self.translations)} languages")
        
    def set_locale(self, locale_code: str) -> None:
        """Set current locale for translations."""
        if locale_code in self.translations:
            self.current_locale = locale_code
            logger.info(f"Locale set to {locale_code}")
        else:
            logger.warning(f"Locale {locale_code} not available, using default")
            
    def translate(self, key: str, locale: Optional[str] = None) -> str:
        """
        Translate a message key to the specified or current locale.
        
        Args:
            key: Translation key
            locale: Target locale (uses current if None)
            
        Returns:
            Translated string or key if translation not found
        """
        target_locale = locale or self.current_locale
        
        if target_locale in self.translations:
            translations = self.translations[target_locale]
            return translations.get(key, key)
        else:
            # Fallback to English
            if "en" in self.translations:
                return self.translations["en"].get(key, key)
            return key
            
    def translate_dict(
        self,
        data: Dict[str, Any],
        key_prefix: str = "",
        locale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate dictionary values that are translation keys.
        
        Args:
            data: Dictionary with translation keys as values
            key_prefix: Prefix to add to keys when looking up translations
            locale: Target locale
            
        Returns:
            Dictionary with translated values
        """
        translated = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                translation_key = f"{key_prefix}.{value}" if key_prefix else value
                translated[key] = self.translate(translation_key, locale)
            elif isinstance(value, dict):
                translated[key] = self.translate_dict(value, key_prefix, locale)
            else:
                translated[key] = value
                
        return translated
        
    def get_available_locales(self) -> List[str]:
        """Get list of available locale codes."""
        return list(self.translations.keys())
        
    def format_sentiment_report(
        self,
        sentiment_data: Dict[str, Any],
        locale: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format sentiment analysis report for specific locale."""
        target_locale = locale or self.current_locale
        
        # Translate sentiment labels
        if "sentiment" in sentiment_data:
            sentiment_scores = sentiment_data["sentiment"]
            translated_scores = {}
            
            for label, score in sentiment_scores.items():
                translation_key = f"sentiment.{label}"
                translated_label = self.translate(translation_key, target_locale)
                translated_scores[translated_label] = score
                
            sentiment_data["sentiment"] = translated_scores
            
        # Translate status messages
        if "status" in sentiment_data:
            status_key = f"status.{sentiment_data['status']}"
            sentiment_data["status_message"] = self.translate(status_key, target_locale)
            
        return sentiment_data


# Global instances
_multilingual_analyzer = None
_localization_manager = None


def get_multilingual_analyzer(**kwargs) -> MultilingualSentimentAnalyzer:
    """Get global multilingual sentiment analyzer instance."""
    global _multilingual_analyzer
    if _multilingual_analyzer is None:
        _multilingual_analyzer = MultilingualSentimentAnalyzer(**kwargs)
    return _multilingual_analyzer


def get_localization_manager(**kwargs) -> LocalizationManager:
    """Get global localization manager instance.""" 
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager(**kwargs)
    return _localization_manager