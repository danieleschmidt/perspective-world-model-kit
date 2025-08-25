"""
Global-first internationalization (i18n) manager for PWMK.
Supports multiple languages, regions, and cultural adaptations.
"""

import os
import json
import gettext
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import locale

from ..utils.logging import LoggingMixin


@dataclass
class LocalizationEntry:
    """Localization entry with metadata."""
    key: str
    value: str
    language: str
    region: Optional[str] = None
    context: Optional[str] = None
    pluralization: Optional[Dict[str, str]] = None
    last_updated: float = 0.0


class I18nManager(LoggingMixin):
    """
    Comprehensive internationalization manager for global PWMK deployment.
    
    Supports:
    - Multiple languages (en, es, fr, de, ja, zh, ar, hi, ru, pt)
    - Regional variations (en-US, en-GB, zh-CN, zh-TW, etc.)
    - Cultural adaptations
    - Pluralization rules
    - Date/time/number formatting
    - Right-to-left (RTL) language support
    """
    
    SUPPORTED_LANGUAGES = {
        'en': {'name': 'English', 'native_name': 'English', 'rtl': False, 'priority': 1},
        'es': {'name': 'Spanish', 'native_name': 'Español', 'rtl': False, 'priority': 2},
        'fr': {'name': 'French', 'native_name': 'Français', 'rtl': False, 'priority': 3},
        'de': {'name': 'German', 'native_name': 'Deutsch', 'rtl': False, 'priority': 4},
        'ja': {'name': 'Japanese', 'native_name': '日本語', 'rtl': False, 'priority': 5},
        'zh': {'name': 'Chinese', 'native_name': '中文', 'rtl': False, 'priority': 6},
        'ar': {'name': 'Arabic', 'native_name': 'العربية', 'rtl': True, 'priority': 7},
        'hi': {'name': 'Hindi', 'native_name': 'हिन्दी', 'rtl': False, 'priority': 8},
        'ru': {'name': 'Russian', 'native_name': 'Русский', 'rtl': False, 'priority': 9},
        'pt': {'name': 'Portuguese', 'native_name': 'Português', 'rtl': False, 'priority': 10},
        'ko': {'name': 'Korean', 'native_name': '한국어', 'rtl': False, 'priority': 11},
        'it': {'name': 'Italian', 'native_name': 'Italiano', 'rtl': False, 'priority': 12},
        'tr': {'name': 'Turkish', 'native_name': 'Türkçe', 'rtl': False, 'priority': 13},
    }
    
    REGIONAL_VARIANTS = {
        'en': ['US', 'GB', 'AU', 'CA', 'IN'],
        'es': ['ES', 'MX', 'AR', 'CO', 'PE'],
        'fr': ['FR', 'CA', 'BE', 'CH'],
        'de': ['DE', 'AT', 'CH'],
        'zh': ['CN', 'TW', 'HK', 'SG'],
        'pt': ['PT', 'BR'],
        'ar': ['SA', 'EG', 'AE', 'MA']
    }
    
    def __init__(self, locales_dir: Optional[str] = None, default_locale: str = "en"):
        super().__init__()
        
        self.locales_dir = Path(locales_dir or "locales")
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Localization data
        self.translations: Dict[str, Dict[str, LocalizationEntry]] = {}
        self.formatters: Dict[str, Dict[str, Any]] = {}
        
        # Initialize localization system
        self._initialize_locales_directory()
        self._load_all_translations()
        self._setup_formatters()
        
        self.log_info(
            f"I18n manager initialized with {len(self.translations)} languages",
            default_locale=default_locale,
            supported_languages=len(self.SUPPORTED_LANGUAGES)
        )
    
    def _initialize_locales_directory(self) -> None:
        """Initialize locales directory structure."""
        self.locales_dir.mkdir(exist_ok=True)
        
        # Create language directories
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            lang_dir = self.locales_dir / lang_code
            lang_dir.mkdir(exist_ok=True)
            
            # Create LC_MESSAGES directory for gettext
            messages_dir = lang_dir / "LC_MESSAGES"
            messages_dir.mkdir(exist_ok=True)
            
            # Create JSON translations file if it doesn't exist
            json_file = lang_dir / "translations.json"
            if not json_file.exists():
                self._create_default_translations(lang_code, json_file)
    
    def _create_default_translations(self, lang_code: str, json_file: Path) -> None:
        """Create default translations for a language."""
        default_translations = {
            # Core PWMK terms
            "pwmk.name": self._get_default_translation("Perspective World Model Kit", lang_code),
            "pwmk.description": self._get_default_translation(
                "Neuro-symbolic AI framework with Theory of Mind capabilities", lang_code
            ),
            
            # UI Elements
            "ui.loading": self._get_default_translation("Loading...", lang_code),
            "ui.save": self._get_default_translation("Save", lang_code),
            "ui.cancel": self._get_default_translation("Cancel", lang_code),
            "ui.confirm": self._get_default_translation("Confirm", lang_code),
            "ui.error": self._get_default_translation("Error", lang_code),
            "ui.success": self._get_default_translation("Success", lang_code),
            "ui.warning": self._get_default_translation("Warning", lang_code),
            
            # AI/ML Terms
            "ai.agent": self._get_default_translation("Agent", lang_code),
            "ai.belief": self._get_default_translation("Belief", lang_code),
            "ai.consciousness": self._get_default_translation("Consciousness", lang_code),
            "ai.intelligence": self._get_default_translation("Intelligence", lang_code),
            "ai.learning": self._get_default_translation("Learning", lang_code),
            "ai.model": self._get_default_translation("Model", lang_code),
            "ai.training": self._get_default_translation("Training", lang_code),
            "ai.prediction": self._get_default_translation("Prediction", lang_code),
            
            # Status Messages
            "status.initializing": self._get_default_translation("Initializing system...", lang_code),
            "status.training": self._get_default_translation("Training in progress...", lang_code),
            "status.completed": self._get_default_translation("Operation completed successfully", lang_code),
            "status.failed": self._get_default_translation("Operation failed", lang_code),
            
            # Error Messages
            "error.network": self._get_default_translation("Network connection error", lang_code),
            "error.auth": self._get_default_translation("Authentication failed", lang_code),
            "error.permission": self._get_default_translation("Permission denied", lang_code),
            "error.not_found": self._get_default_translation("Resource not found", lang_code),
            
            # Time formats
            "time.now": self._get_default_translation("now", lang_code),
            "time.minutes_ago": self._get_default_translation("{count} minutes ago", lang_code),
            "time.hours_ago": self._get_default_translation("{count} hours ago", lang_code),
            "time.days_ago": self._get_default_translation("{count} days ago", lang_code),
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
    
    def _get_default_translation(self, text: str, lang_code: str) -> str:
        """Get default translation for text in specified language."""
        # Simple translations for demonstration
        translations = {
            'es': {
                'Loading...': 'Cargando...',
                'Save': 'Guardar',
                'Cancel': 'Cancelar',
                'Confirm': 'Confirmar',
                'Error': 'Error',
                'Success': 'Éxito',
                'Warning': 'Advertencia',
                'Agent': 'Agente',
                'Belief': 'Creencia',
                'Consciousness': 'Conciencia',
                'Intelligence': 'Inteligencia',
                'Learning': 'Aprendizaje',
                'Model': 'Modelo',
                'Training': 'Entrenamiento',
                'Prediction': 'Predicción',
                'Perspective World Model Kit': 'Kit de Modelo del Mundo con Perspectiva',
                'Neuro-symbolic AI framework with Theory of Mind capabilities': 'Framework de IA neuro-simbólica con capacidades de Teoría de la Mente'
            },
            'fr': {
                'Loading...': 'Chargement...',
                'Save': 'Sauvegarder',
                'Cancel': 'Annuler',
                'Confirm': 'Confirmer',
                'Error': 'Erreur',
                'Success': 'Succès',
                'Warning': 'Avertissement',
                'Agent': 'Agent',
                'Belief': 'Croyance',
                'Consciousness': 'Conscience',
                'Intelligence': 'Intelligence',
                'Learning': 'Apprentissage',
                'Model': 'Modèle',
                'Training': 'Entraînement',
                'Prediction': 'Prédiction',
                'Perspective World Model Kit': 'Kit de Modèle du Monde avec Perspective',
                'Neuro-symbolic AI framework with Theory of Mind capabilities': 'Framework d\'IA neuro-symbolique avec capacités de Théorie de l\'Esprit'
            },
            'de': {
                'Loading...': 'Laden...',
                'Save': 'Speichern',
                'Cancel': 'Abbrechen',
                'Confirm': 'Bestätigen',
                'Error': 'Fehler',
                'Success': 'Erfolg',
                'Warning': 'Warnung',
                'Agent': 'Agent',
                'Belief': 'Glaube',
                'Consciousness': 'Bewusstsein',
                'Intelligence': 'Intelligenz',
                'Learning': 'Lernen',
                'Model': 'Modell',
                'Training': 'Training',
                'Prediction': 'Vorhersage',
                'Perspective World Model Kit': 'Perspektiven-Weltmodell-Kit',
                'Neuro-symbolic AI framework with Theory of Mind capabilities': 'Neuro-symbolisches KI-Framework mit Theory-of-Mind-Fähigkeiten'
            },
            'ja': {
                'Loading...': '読み込み中...',
                'Save': '保存',
                'Cancel': 'キャンセル',
                'Confirm': '確認',
                'Error': 'エラー',
                'Success': '成功',
                'Warning': '警告',
                'Agent': 'エージェント',
                'Belief': '信念',
                'Consciousness': '意識',
                'Intelligence': '知能',
                'Learning': '学習',
                'Model': 'モデル',
                'Training': '訓練',
                'Prediction': '予測',
                'Perspective World Model Kit': '視点世界モデルキット',
                'Neuro-symbolic AI framework with Theory of Mind capabilities': '心の理論機能を持つニューロシンボリックAIフレームワーク'
            },
            'zh': {
                'Loading...': '加载中...',
                'Save': '保存',
                'Cancel': '取消',
                'Confirm': '确认',
                'Error': '错误',
                'Success': '成功',
                'Warning': '警告',
                'Agent': '代理',
                'Belief': '信念',
                'Consciousness': '意识',
                'Intelligence': '智能',
                'Learning': '学习',
                'Model': '模型',
                'Training': '训练',
                'Prediction': '预测',
                'Perspective World Model Kit': '透视世界模型工具包',
                'Neuro-symbolic AI framework with Theory of Mind capabilities': '具有心智理论能力的神经符号AI框架'
            }
        }
        
        if lang_code in translations and text in translations[lang_code]:
            return translations[lang_code][text]
        
        return text  # Fallback to original text
    
    def _load_all_translations(self) -> None:
        """Load all available translations."""
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            self._load_language_translations(lang_code)
    
    def _load_language_translations(self, lang_code: str) -> None:
        """Load translations for a specific language."""
        lang_dir = self.locales_dir / lang_code
        json_file = lang_dir / "translations.json"
        
        if not json_file.exists():
            self.log_warning(f"No translations file found for {lang_code}")
            return
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                translations_data = json.load(f)
            
            self.translations[lang_code] = {}
            
            for key, value in translations_data.items():
                self.translations[lang_code][key] = LocalizationEntry(
                    key=key,
                    value=value,
                    language=lang_code,
                    last_updated=json_file.stat().st_mtime
                )
            
            self.log_info(
                f"Loaded {len(translations_data)} translations for {lang_code}"
            )
            
        except Exception as e:
            self.log_error(f"Failed to load translations for {lang_code}: {str(e)}")
    
    def _setup_formatters(self) -> None:
        """Setup locale-specific formatters."""
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            self.formatters[lang_code] = {
                'date_format': self._get_date_format(lang_code),
                'time_format': self._get_time_format(lang_code),
                'number_format': self._get_number_format(lang_code),
                'currency_format': self._get_currency_format(lang_code)
            }
    
    def _get_date_format(self, lang_code: str) -> str:
        """Get date format for language."""
        formats = {
            'en': '%m/%d/%Y',
            'es': '%d/%m/%Y',
            'fr': '%d/%m/%Y',
            'de': '%d.%m.%Y',
            'ja': '%Y年%m月%d日',
            'zh': '%Y年%m月%d日',
            'ar': '%d/%m/%Y',
            'hi': '%d/%m/%Y',
            'ru': '%d.%m.%Y',
            'pt': '%d/%m/%Y',
            'ko': '%Y년 %m월 %d일',
            'it': '%d/%m/%Y',
            'tr': '%d.%m.%Y'
        }
        return formats.get(lang_code, '%Y-%m-%d')
    
    def _get_time_format(self, lang_code: str) -> str:
        """Get time format for language."""
        formats = {
            'en': '%I:%M %p',
            'es': '%H:%M',
            'fr': '%H:%M',
            'de': '%H:%M',
            'ja': '%H時%M分',
            'zh': '%H:%M',
            'ar': '%H:%M',
            'hi': '%H:%M',
            'ru': '%H:%M',
            'pt': '%H:%M',
            'ko': '%H시 %M분',
            'it': '%H:%M',
            'tr': '%H:%M'
        }
        return formats.get(lang_code, '%H:%M')
    
    def _get_number_format(self, lang_code: str) -> Dict[str, str]:
        """Get number formatting for language."""
        formats = {
            'en': {'decimal': '.', 'thousands': ','},
            'es': {'decimal': ',', 'thousands': '.'},
            'fr': {'decimal': ',', 'thousands': ' '},
            'de': {'decimal': ',', 'thousands': '.'},
            'ja': {'decimal': '.', 'thousands': ','},
            'zh': {'decimal': '.', 'thousands': ','},
            'ar': {'decimal': '.', 'thousands': ','},
            'hi': {'decimal': '.', 'thousands': ','},
            'ru': {'decimal': ',', 'thousands': ' '},
            'pt': {'decimal': ',', 'thousands': '.'},
            'ko': {'decimal': '.', 'thousands': ','},
            'it': {'decimal': ',', 'thousands': '.'},
            'tr': {'decimal': ',', 'thousands': '.'}
        }
        return formats.get(lang_code, {'decimal': '.', 'thousands': ','})
    
    def _get_currency_format(self, lang_code: str) -> Dict[str, str]:
        """Get currency formatting for language."""
        formats = {
            'en': {'symbol': '$', 'position': 'before'},
            'es': {'symbol': '€', 'position': 'after'},
            'fr': {'symbol': '€', 'position': 'after'},
            'de': {'symbol': '€', 'position': 'after'},
            'ja': {'symbol': '¥', 'position': 'before'},
            'zh': {'symbol': '¥', 'position': 'before'},
            'ar': {'symbol': 'ر.س', 'position': 'after'},
            'hi': {'symbol': '₹', 'position': 'before'},
            'ru': {'symbol': '₽', 'position': 'after'},
            'pt': {'symbol': '€', 'position': 'after'},
            'ko': {'symbol': '₩', 'position': 'before'},
            'it': {'symbol': '€', 'position': 'after'},
            'tr': {'symbol': '₺', 'position': 'after'}
        }
        return formats.get(lang_code, {'symbol': '$', 'position': 'before'})
    
    def set_locale(self, locale_code: str) -> bool:
        """Set current locale."""
        # Parse locale code (e.g., "en-US", "zh-CN")
        if '-' in locale_code:
            lang_code, region_code = locale_code.split('-', 1)
        else:
            lang_code = locale_code
            region_code = None
        
        if lang_code not in self.SUPPORTED_LANGUAGES:
            self.log_warning(f"Unsupported language: {lang_code}")
            return False
        
        self.current_locale = locale_code
        
        self.log_info(
            f"Locale changed to {locale_code}",
            language=lang_code,
            region=region_code
        )
        
        return True
    
    def get_translation(
        self,
        key: str,
        locale: Optional[str] = None,
        default: Optional[str] = None,
        **kwargs
    ) -> str:
        """Get translation for a key."""
        target_locale = locale or self.current_locale
        
        # Parse locale
        if '-' in target_locale:
            lang_code = target_locale.split('-')[0]
        else:
            lang_code = target_locale
        
        # Try to get translation
        if (lang_code in self.translations and 
            key in self.translations[lang_code]):
            translation = self.translations[lang_code][key].value
        elif (self.default_locale in self.translations and
              key in self.translations[self.default_locale]):
            translation = self.translations[self.default_locale][key].value
        else:
            translation = default or key
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.log_warning(f"Translation formatting failed for {key}: {str(e)}")
        
        return translation
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format date according to locale."""
        target_locale = locale or self.current_locale
        lang_code = target_locale.split('-')[0]
        
        date_format = self.formatters.get(lang_code, {}).get('date_format', '%Y-%m-%d')
        return date.strftime(date_format)
    
    def format_time(self, time: datetime, locale: Optional[str] = None) -> str:
        """Format time according to locale."""
        target_locale = locale or self.current_locale
        lang_code = target_locale.split('-')[0]
        
        time_format = self.formatters.get(lang_code, {}).get('time_format', '%H:%M')
        return time.strftime(time_format)
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale."""
        target_locale = locale or self.current_locale
        lang_code = target_locale.split('-')[0]
        
        format_info = self.formatters.get(lang_code, {}).get('number_format', {})
        decimal_sep = format_info.get('decimal', '.')
        thousands_sep = format_info.get('thousands', ',')
        
        # Simple number formatting
        str_number = f"{number:.2f}"
        integer_part, decimal_part = str_number.split('.')
        
        # Add thousands separators
        if len(integer_part) > 3:
            formatted_integer = ""
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_integer = thousands_sep + formatted_integer
                formatted_integer = digit + formatted_integer
        else:
            formatted_integer = integer_part
        
        return f"{formatted_integer}{decimal_sep}{decimal_part}"
    
    def format_currency(self, amount: float, locale: Optional[str] = None) -> str:
        """Format currency according to locale."""
        target_locale = locale or self.current_locale
        lang_code = target_locale.split('-')[0]
        
        format_info = self.formatters.get(lang_code, {}).get('currency_format', {})
        symbol = format_info.get('symbol', '$')
        position = format_info.get('position', 'before')
        
        formatted_amount = self.format_number(amount, locale)
        
        if position == 'before':
            return f"{symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {symbol}"
    
    def is_rtl(self, locale: Optional[str] = None) -> bool:
        """Check if locale uses right-to-left text direction."""
        target_locale = locale or self.current_locale
        lang_code = target_locale.split('-')[0]
        
        return self.SUPPORTED_LANGUAGES.get(lang_code, {}).get('rtl', False)
    
    def get_available_locales(self) -> List[Dict[str, Any]]:
        """Get list of available locales."""
        locales = []
        
        for lang_code, lang_info in self.SUPPORTED_LANGUAGES.items():
            # Base language
            locales.append({
                'code': lang_code,
                'name': lang_info['name'],
                'native_name': lang_info['native_name'],
                'rtl': lang_info['rtl'],
                'region': None,
                'available': lang_code in self.translations
            })
            
            # Regional variants
            if lang_code in self.REGIONAL_VARIANTS:
                for region in self.REGIONAL_VARIANTS[lang_code]:
                    locale_code = f"{lang_code}-{region}"
                    locales.append({
                        'code': locale_code,
                        'name': f"{lang_info['name']} ({region})",
                        'native_name': lang_info['native_name'],
                        'rtl': lang_info['rtl'],
                        'region': region,
                        'available': lang_code in self.translations
                    })
        
        return sorted(locales, key=lambda x: self.SUPPORTED_LANGUAGES[x['code'].split('-')[0]]['priority'])
    
    def export_translations(self, lang_code: str) -> Dict[str, Any]:
        """Export translations for a language."""
        if lang_code not in self.translations:
            return {}
        
        return {
            key: entry.value
            for key, entry in self.translations[lang_code].items()
        }
    
    def get_localization_stats(self) -> Dict[str, Any]:
        """Get localization statistics."""
        stats = {
            'total_languages': len(self.SUPPORTED_LANGUAGES),
            'loaded_languages': len(self.translations),
            'current_locale': self.current_locale,
            'rtl_languages': len([l for l in self.SUPPORTED_LANGUAGES.values() if l['rtl']]),
            'translation_keys': {},
            'completion_rates': {}
        }
        
        # Calculate translation key counts and completion rates
        if self.default_locale in self.translations:
            total_keys = len(self.translations[self.default_locale])
            
            for lang_code, translations in self.translations.items():
                key_count = len(translations)
                stats['translation_keys'][lang_code] = key_count
                
                if total_keys > 0:
                    completion_rate = (key_count / total_keys) * 100
                    stats['completion_rates'][lang_code] = completion_rate
        
        return stats


# Global i18n manager instance
global_i18n = I18nManager()


def t(key: str, locale: Optional[str] = None, default: Optional[str] = None, **kwargs) -> str:
    """Shorthand function for getting translations."""
    return global_i18n.get_translation(key, locale=locale, default=default, **kwargs)


def set_global_locale(locale_code: str) -> bool:
    """Set global locale."""
    return global_i18n.set_locale(locale_code)


def get_current_locale() -> str:
    """Get current global locale."""
    return global_i18n.current_locale