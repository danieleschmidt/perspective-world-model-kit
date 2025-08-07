"""
Regional deployment and configuration support for sentiment analysis.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import logging
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2" 
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    BRAZIL = "sa-east-1"
    INDIA = "ap-south-1"


class ComplianceRegime(Enum):
    """Data protection and compliance regimes."""
    GDPR = "gdpr"           # European Union General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    PDPA_SG = "pdpa_sg"     # Singapore Personal Data Protection Act
    PDPA_TH = "pdpa_th"     # Thailand Personal Data Protection Act
    LGPD = "lgpd"           # Brazil Lei Geral de Proteção de Dados
    PIPEDA = "pipeda"       # Canada Personal Information Protection and Electronic Documents Act
    PRIVACY_ACT = "privacy_act_au"  # Australia Privacy Act


@dataclass
class RegionalConfig:
    """Configuration for a specific region."""
    region: Region
    compliance_regime: List[ComplianceRegime]
    data_residency_required: bool
    cross_border_transfers_allowed: bool
    encryption_requirements: Dict[str, Any]
    retention_limits: Dict[str, int]  # in days
    supported_languages: List[str]
    local_model_endpoints: Dict[str, str]
    timezone: str
    currency: str
    date_format: str
    

class RegionalConfigManager:
    """
    Manages regional configurations for multi-region deployments.
    """
    
    def __init__(self):
        self.regional_configs: Dict[Region, RegionalConfig] = {}
        self._initialize_default_configs()
        
    def _initialize_default_configs(self) -> None:
        """Initialize default regional configurations."""
        configs = {
            Region.US_EAST: RegionalConfig(
                region=Region.US_EAST,
                compliance_regime=[ComplianceRegime.CCPA],
                data_residency_required=False,
                cross_border_transfers_allowed=True,
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS-1.3",
                    "key_management": "AWS-KMS"
                },
                retention_limits={
                    "sentiment_data": 365,
                    "personal_data": 1095,  # 3 years
                    "communication_logs": 180
                },
                supported_languages=["en", "es"],
                local_model_endpoints={
                    "sentiment_analysis": "https://api-us-east.sentiment.ai/v1/analyze",
                    "language_detection": "https://api-us-east.sentiment.ai/v1/detect"
                },
                timezone="America/New_York",
                currency="USD",
                date_format="%m/%d/%Y"
            ),
            
            Region.EU_WEST: RegionalConfig(
                region=Region.EU_WEST,
                compliance_regime=[ComplianceRegime.GDPR],
                data_residency_required=True,
                cross_border_transfers_allowed=False,
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS-1.3",
                    "key_management": "AWS-KMS-EU",
                    "pseudonymization": "required"
                },
                retention_limits={
                    "sentiment_data": 180,   # Shorter for GDPR
                    "personal_data": 730,    # 2 years max
                    "communication_logs": 90
                },
                supported_languages=["en", "fr", "de", "es", "it"],
                local_model_endpoints={
                    "sentiment_analysis": "https://api-eu-west.sentiment.ai/v1/analyze",
                    "language_detection": "https://api-eu-west.sentiment.ai/v1/detect"
                },
                timezone="Europe/London",
                currency="EUR",
                date_format="%d/%m/%Y"
            ),
            
            Region.EU_CENTRAL: RegionalConfig(
                region=Region.EU_CENTRAL,
                compliance_regime=[ComplianceRegime.GDPR],
                data_residency_required=True,
                cross_border_transfers_allowed=False,
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS-1.3",
                    "key_management": "AWS-KMS-EU",
                    "pseudonymization": "required"
                },
                retention_limits={
                    "sentiment_data": 180,
                    "personal_data": 730,
                    "communication_logs": 90
                },
                supported_languages=["de", "en", "fr"],
                local_model_endpoints={
                    "sentiment_analysis": "https://api-eu-central.sentiment.ai/v1/analyze",
                    "language_detection": "https://api-eu-central.sentiment.ai/v1/detect"
                },
                timezone="Europe/Berlin",
                currency="EUR", 
                date_format="%d.%m.%Y"
            ),
            
            Region.ASIA_PACIFIC: RegionalConfig(
                region=Region.ASIA_PACIFIC,
                compliance_regime=[ComplianceRegime.PDPA_SG],
                data_residency_required=True,
                cross_border_transfers_allowed=True,  # With adequate safeguards
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS-1.3", 
                    "key_management": "AWS-KMS-AP"
                },
                retention_limits={
                    "sentiment_data": 365,
                    "personal_data": 1095,
                    "communication_logs": 180
                },
                supported_languages=["en", "zh", "ja"],
                local_model_endpoints={
                    "sentiment_analysis": "https://api-ap-southeast.sentiment.ai/v1/analyze",
                    "language_detection": "https://api-ap-southeast.sentiment.ai/v1/detect"
                },
                timezone="Asia/Singapore",
                currency="USD",
                date_format="%d/%m/%Y"
            ),
            
            Region.ASIA_NORTHEAST: RegionalConfig(
                region=Region.ASIA_NORTHEAST,
                compliance_regime=[],  # Japan-specific privacy laws
                data_residency_required=False,
                cross_border_transfers_allowed=True,
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS-1.3",
                    "key_management": "AWS-KMS-AP"
                },
                retention_limits={
                    "sentiment_data": 365,
                    "personal_data": 1095,
                    "communication_logs": 180
                },
                supported_languages=["ja", "en"],
                local_model_endpoints={
                    "sentiment_analysis": "https://api-ap-northeast.sentiment.ai/v1/analyze",
                    "language_detection": "https://api-ap-northeast.sentiment.ai/v1/detect"
                },
                timezone="Asia/Tokyo",
                currency="JPY",
                date_format="%Y/%m/%d"
            ),
            
            Region.BRAZIL: RegionalConfig(
                region=Region.BRAZIL,
                compliance_regime=[ComplianceRegime.LGPD],
                data_residency_required=True,
                cross_border_transfers_allowed=False,
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS-1.3",
                    "key_management": "AWS-KMS-SA",
                    "anonymization": "required"
                },
                retention_limits={
                    "sentiment_data": 180,
                    "personal_data": 365,
                    "communication_logs": 90
                },
                supported_languages=["pt", "en"],
                local_model_endpoints={
                    "sentiment_analysis": "https://api-sa-east.sentiment.ai/v1/analyze",
                    "language_detection": "https://api-sa-east.sentiment.ai/v1/detect"
                },
                timezone="America/Sao_Paulo",
                currency="BRL",
                date_format="%d/%m/%Y"
            )
        }
        
        self.regional_configs.update(configs)
        logger.info(f"Initialized configurations for {len(configs)} regions")
        
    def get_config(self, region: Region) -> RegionalConfig:
        """Get configuration for specific region."""
        if region not in self.regional_configs:
            logger.warning(f"No configuration for region {region}, using US_EAST default")
            return self.regional_configs[Region.US_EAST]
        return self.regional_configs[region]
        
    def get_compliance_requirements(self, region: Region) -> List[ComplianceRegime]:
        """Get compliance requirements for region."""
        config = self.get_config(region)
        return config.compliance_regime
        
    def can_transfer_data(self, source_region: Region, target_region: Region) -> bool:
        """Check if data transfer between regions is allowed."""
        source_config = self.get_config(source_region)
        target_config = self.get_config(target_region)
        
        # Check if source allows cross-border transfers
        if not source_config.cross_border_transfers_allowed:
            logger.warning(f"Cross-border transfers not allowed from {source_region}")
            return False
            
        # Check compliance compatibility
        source_regimes = set(source_config.compliance_regime)
        target_regimes = set(target_config.compliance_regime)
        
        # GDPR has strict requirements
        if ComplianceRegime.GDPR in source_regimes:
            # Can only transfer to regions with GDPR or adequacy decisions
            adequate_regimes = {
                ComplianceRegime.GDPR,
                ComplianceRegime.PIPEDA  # Canada has adequacy decision
            }
            if not target_regimes.intersection(adequate_regimes):
                logger.warning(f"GDPR compliance prevents transfer from {source_region} to {target_region}")
                return False
                
        return True
        
    def get_encryption_config(self, region: Region) -> Dict[str, Any]:
        """Get encryption requirements for region."""
        config = self.get_config(region)
        return config.encryption_requirements
        
    def get_retention_limit(self, region: Region, data_type: str) -> int:
        """Get retention limit for specific data type in region."""
        config = self.get_config(region)
        return config.retention_limits.get(data_type, 365)  # Default 1 year
        
    def get_supported_languages(self, region: Region) -> List[str]:
        """Get supported languages for region.""" 
        config = self.get_config(region)
        return config.supported_languages
        
    def get_model_endpoint(self, region: Region, service: str) -> Optional[str]:
        """Get regional model endpoint for service."""
        config = self.get_config(region)
        return config.local_model_endpoints.get(service)
        

class RegionalSentimentAnalyzer:
    """
    Sentiment analyzer with regional compliance and configuration support.
    """
    
    def __init__(
        self,
        region: Region,
        config_manager: Optional[RegionalConfigManager] = None
    ):
        self.region = region
        self.config_manager = config_manager or RegionalConfigManager()
        self.config = self.config_manager.get_config(region)
        
        # Initialize compliance controller based on regional requirements
        self._initialize_compliance_controller()
        
        # Initialize regional sentiment analyzer
        self._initialize_regional_analyzer()
        
        logger.info(f"Regional sentiment analyzer initialized for {region.value}")
        
    def _initialize_compliance_controller(self) -> None:
        """Initialize compliance controller based on regional requirements."""
        from .compliance import PrivacyController, LegalBasis
        
        # Configure based on regional compliance regimes
        auto_anonymize = False
        default_retention_days = 365
        
        if ComplianceRegime.GDPR in self.config.compliance_regime:
            auto_anonymize = True
            default_retention_days = 180  # Shorter retention for GDPR
        elif ComplianceRegime.LGPD in self.config.compliance_regime:
            auto_anonymize = True
            default_retention_days = 180
            
        self.privacy_controller = PrivacyController(
            data_controller_name=f"PWMK-{self.region.value}",
            default_retention_days=default_retention_days,
            auto_anonymize=auto_anonymize
        )
        
    def _initialize_regional_analyzer(self) -> None:
        """Initialize region-specific sentiment analyzer."""
        from .i18n import MultilingualSentimentAnalyzer
        
        supported_languages = self.config.supported_languages
        
        self.multilingual_analyzer = MultilingualSentimentAnalyzer(
            default_language=supported_languages[0] if supported_languages else "en",
            supported_languages=supported_languages
        )
        
    def analyze_with_compliance(
        self,
        text: str,
        user_id: str,
        language: Optional[str] = None,
        consent_given: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sentiment with full regional compliance.
        
        Args:
            text: Text to analyze
            user_id: User identifier
            language: Language code (auto-detected if None)
            consent_given: Whether user has given explicit consent
            
        Returns:
            Sentiment analysis results with compliance metadata
        """
        # Determine legal basis based on regional requirements
        from .compliance import LegalBasis, ProcessingPurpose, DataCategory
        
        legal_basis = LegalBasis.CONSENT if consent_given else LegalBasis.LEGITIMATE_INTEREST
        
        # For GDPR regions, require explicit consent for sentiment analysis
        if ComplianceRegime.GDPR in self.config.compliance_regime and not consent_given:
            return {
                "error": "explicit_consent_required",
                "message": "Explicit consent required for sentiment analysis under GDPR",
                "region": self.region.value
            }
            
        # Register data subject if not already registered
        try:
            subject_id = self.privacy_controller.register_data_subject(
                original_id=user_id,
                legal_basis=legal_basis,
                data_categories={
                    DataCategory.COMMUNICATION_CONTENT,
                    DataCategory.SENTIMENT_DATA,
                    DataCategory.BEHAVIORAL_DATA
                },
                consent_given=consent_given
            )
        except ValueError:
            # Subject already registered
            subject_id = self.privacy_controller._pseudonymize_id(user_id)
            
        # Record processing activity
        compliance_result = self.privacy_controller.process_sentiment_analysis(
            subject_id=subject_id,
            text_data=text,
            processor=f"regional-analyzer-{self.region.value}"
        )
        
        # Perform sentiment analysis
        analysis_result = self.multilingual_analyzer.analyze_text(
            text=text,
            language=language,
            return_language_info=True
        )
        
        # Combine results
        result = {
            **analysis_result,
            "compliance": {
                "region": self.region.value,
                "compliance_regimes": [regime.value for regime in self.config.compliance_regime],
                "processing_record_id": compliance_result["processing_record_id"],
                "legal_basis": compliance_result["legal_basis"],
                "data_anonymized": compliance_result["data_anonymized"],
                "retention_until": compliance_result["retention_until"],
                "data_residency": self.config.data_residency_required
            }
        }
        
        return result
        
    def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights request with regional compliance."""
        result = self.privacy_controller.handle_data_subject_request(
            original_id=user_id,
            request_type=request_type,
            additional_data=additional_data
        )
        
        # Add regional context
        result["region"] = self.region.value
        result["compliance_regimes"] = [regime.value for regime in self.config.compliance_regime]
        
        return result
        
    def get_regional_status(self) -> Dict[str, Any]:
        """Get status information for the regional deployment."""
        compliance_report = self.privacy_controller.get_compliance_report()
        
        status = {
            "region": self.region.value,
            "timezone": self.config.timezone,
            "compliance_regimes": [regime.value for regime in self.config.compliance_regime],
            "supported_languages": self.config.supported_languages,
            "data_residency_required": self.config.data_residency_required,
            "cross_border_transfers_allowed": self.config.cross_border_transfers_allowed,
            "encryption_requirements": self.config.encryption_requirements,
            "compliance_summary": compliance_report["summary"],
            "model_endpoints": self.config.local_model_endpoints
        }
        
        return status
        
    def validate_cross_region_transfer(self, target_region: Region) -> Dict[str, Any]:
        """Validate if data transfer to target region is allowed."""
        allowed = self.config_manager.can_transfer_data(self.region, target_region)
        
        target_config = self.config_manager.get_config(target_region)
        
        result = {
            "source_region": self.region.value,
            "target_region": target_region.value,
            "transfer_allowed": allowed,
            "source_regimes": [regime.value for regime in self.config.compliance_regime],
            "target_regimes": [regime.value for regime in target_config.compliance_regime]
        }
        
        if not allowed:
            result["reasons"] = self._get_transfer_restriction_reasons(target_region)
            
        return result
        
    def _get_transfer_restriction_reasons(self, target_region: Region) -> List[str]:
        """Get reasons why cross-region transfer is restricted."""
        reasons = []
        
        if not self.config.cross_border_transfers_allowed:
            reasons.append("Source region does not allow cross-border transfers")
            
        if ComplianceRegime.GDPR in self.config.compliance_regime:
            target_config = self.config_manager.get_config(target_region)
            if ComplianceRegime.GDPR not in target_config.compliance_regime:
                reasons.append("GDPR compliance requires adequate level of protection in target region")
                
        if self.config.data_residency_required:
            reasons.append("Data residency requirements prevent cross-border transfer")
            
        return reasons
        
    def cleanup_expired_data(self) -> Dict[str, Any]:
        """Clean up expired data according to regional retention policies."""
        cleanup_result = self.privacy_controller.cleanup_expired_data()
        
        cleanup_result.update({
            "region": self.region.value,
            "retention_policies": self.config.retention_limits
        })
        
        return cleanup_result


def get_optimal_region(
    user_location: Optional[str] = None,
    preferred_language: Optional[str] = None,
    compliance_requirements: Optional[List[str]] = None
) -> Region:
    """
    Determine optimal region for deployment based on user requirements.
    
    Args:
        user_location: User's geographic location
        preferred_language: User's preferred language
        compliance_requirements: Required compliance regimes
        
    Returns:
        Recommended region for deployment
    """
    config_manager = RegionalConfigManager()
    
    # Map locations to regions
    location_mapping = {
        "US": [Region.US_EAST, Region.US_WEST],
        "CA": [Region.CANADA],
        "UK": [Region.EU_WEST],
        "DE": [Region.EU_CENTRAL],
        "FR": [Region.EU_WEST],
        "SG": [Region.ASIA_PACIFIC],
        "JP": [Region.ASIA_NORTHEAST],
        "AU": [Region.AUSTRALIA],
        "BR": [Region.BRAZIL]
    }
    
    # Map compliance requirements to regions
    compliance_mapping = {
        "gdpr": [Region.EU_WEST, Region.EU_CENTRAL],
        "ccpa": [Region.US_EAST, Region.US_WEST],
        "pdpa_sg": [Region.ASIA_PACIFIC],
        "lgpd": [Region.BRAZIL]
    }
    
    candidate_regions = list(Region)
    
    # Filter by location
    if user_location:
        location_regions = location_mapping.get(user_location.upper(), [])
        if location_regions:
            candidate_regions = [r for r in candidate_regions if r in location_regions]
            
    # Filter by compliance requirements
    if compliance_requirements:
        compliance_regions = set()
        for req in compliance_requirements:
            compliance_regions.update(compliance_mapping.get(req.lower(), []))
        if compliance_regions:
            candidate_regions = [r for r in candidate_regions if r in compliance_regions]
            
    # Filter by language support
    if preferred_language:
        language_compatible_regions = []
        for region in candidate_regions:
            config = config_manager.get_config(region)
            if preferred_language in config.supported_languages:
                language_compatible_regions.append(region)
        if language_compatible_regions:
            candidate_regions = language_compatible_regions
            
    # Return first candidate or default to US_EAST
    return candidate_regions[0] if candidate_regions else Region.US_EAST


# Global instances
_config_manager = None


def get_regional_config_manager() -> RegionalConfigManager:
    """Get global regional configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = RegionalConfigManager()
    return _config_manager