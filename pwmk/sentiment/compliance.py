"""
Data privacy and compliance features for sentiment analysis.
"""

import hashlib
import uuid
import time
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataCategory(Enum):
    """Categories of personal data as defined by privacy regulations."""
    PERSONAL_IDENTIFIERS = "personal_identifiers"  # Names, emails, phone numbers
    DEMOGRAPHIC_DATA = "demographic_data"          # Age, gender, location
    BEHAVIORAL_DATA = "behavioral_data"            # Communication patterns, preferences
    SENTIMENT_DATA = "sentiment_data"              # Emotional state, opinions
    BIOMETRIC_DATA = "biometric_data"              # Voice patterns, writing style
    COMMUNICATION_CONTENT = "communication_content" # Message text, conversations


class ProcessingPurpose(Enum):
    """Lawful purposes for data processing."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    BEHAVIOR_MODELING = "behavior_modeling" 
    SYSTEM_OPTIMIZATION = "system_optimization"
    RESEARCH_ANALYSIS = "research_analysis"
    COMPLIANCE_MONITORING = "compliance_monitoring"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"                    # Article 6(1)(a)
    CONTRACT = "contract"                  # Article 6(1)(b)  
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"    # Article 6(1)(d)
    PUBLIC_TASK = "public_task"           # Article 6(1)(e)
    LEGITIMATE_INTEREST = "legitimate_interest"  # Article 6(1)(f)


@dataclass
class DataSubject:
    """Represents a data subject (individual) whose data is being processed."""
    subject_id: str  # Pseudonymized identifier
    real_id: Optional[str] = None  # Original identifier (encrypted/hashed)
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    legal_basis: Optional[LegalBasis] = None
    data_categories: Set[DataCategory] = field(default_factory=set)
    retention_period: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    
@dataclass 
class ProcessingRecord:
    """Record of data processing activity for compliance tracking."""
    record_id: str
    subject_id: str
    processing_purpose: ProcessingPurpose
    data_categories: Set[DataCategory]
    legal_basis: LegalBasis
    data_controller: str
    data_processor: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retention_until: Optional[datetime] = None
    anonymized: bool = False
    
    
class PrivacyController:
    """
    Privacy controller for managing data protection compliance.
    Handles GDPR, CCPA, PDPA compliance requirements.
    """
    
    def __init__(
        self,
        data_controller_name: str,
        default_retention_days: int = 365,
        auto_anonymize: bool = True
    ):
        self.data_controller_name = data_controller_name
        self.default_retention_days = default_retention_days
        self.auto_anonymize = auto_anonymize
        
        # Storage for compliance data
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: Dict[str, ProcessingRecord] = {}
        self.consent_records: Dict[str, Dict] = {}
        
        # Anonymization mapping (for right to be forgotten)
        self.anonymization_map: Dict[str, str] = {}
        
        logger.info(f"Privacy controller initialized for {data_controller_name}")
        
    def register_data_subject(
        self,
        original_id: str,
        legal_basis: LegalBasis,
        data_categories: Set[DataCategory],
        consent_given: bool = False,
        retention_period: Optional[timedelta] = None
    ) -> str:
        """
        Register a new data subject for processing.
        
        Args:
            original_id: Original identifier (will be pseudonymized)
            legal_basis: Legal basis for processing
            data_categories: Categories of data to be processed
            consent_given: Whether explicit consent was given
            retention_period: Custom retention period
            
        Returns:
            Pseudonymized subject ID
        """
        # Create pseudonymized identifier
        subject_id = self._pseudonymize_id(original_id)
        
        # Store encrypted original ID if needed for data subject rights
        encrypted_original_id = self._encrypt_id(original_id)
        
        data_subject = DataSubject(
            subject_id=subject_id,
            real_id=encrypted_original_id,
            consent_given=consent_given,
            consent_timestamp=datetime.utcnow() if consent_given else None,
            legal_basis=legal_basis,
            data_categories=data_categories,
            retention_period=retention_period or timedelta(days=self.default_retention_days)
        )
        
        self.data_subjects[subject_id] = data_subject
        
        logger.info(f"Registered data subject {subject_id} with legal basis {legal_basis.value}")
        
        return subject_id
        
    def record_processing_activity(
        self,
        subject_id: str,
        processing_purpose: ProcessingPurpose,
        data_categories: Set[DataCategory],
        data_processor: Optional[str] = None
    ) -> str:
        """
        Record a data processing activity.
        
        Args:
            subject_id: Pseudonymized subject identifier
            processing_purpose: Purpose of processing
            data_categories: Categories of data processed
            data_processor: Third party processor (if any)
            
        Returns:
            Processing record ID
        """
        if subject_id not in self.data_subjects:
            raise ValueError(f"Data subject {subject_id} not registered")
            
        data_subject = self.data_subjects[subject_id]
        
        # Verify we have legal basis for these data categories
        if not data_categories.issubset(data_subject.data_categories):
            unauthorized_categories = data_categories - data_subject.data_categories
            raise ValueError(f"No legal basis for processing categories: {unauthorized_categories}")
            
        record_id = str(uuid.uuid4())
        
        # Calculate retention period
        retention_until = None
        if data_subject.retention_period:
            retention_until = datetime.utcnow() + data_subject.retention_period
            
        processing_record = ProcessingRecord(
            record_id=record_id,
            subject_id=subject_id,
            processing_purpose=processing_purpose,
            data_categories=data_categories,
            legal_basis=data_subject.legal_basis,
            data_controller=self.data_controller_name,
            data_processor=data_processor,
            retention_until=retention_until
        )
        
        self.processing_records[record_id] = processing_record
        
        logger.debug(f"Recorded processing activity {record_id} for subject {subject_id}")
        
        return record_id
        
    def process_sentiment_analysis(
        self,
        subject_id: str,
        text_data: str,
        processor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process sentiment analysis with full compliance tracking.
        
        Args:
            subject_id: Data subject identifier
            text_data: Text to analyze
            processor: Name of processing system
            
        Returns:
            Processing result with compliance metadata
        """
        # Record the processing activity
        record_id = self.record_processing_activity(
            subject_id=subject_id,
            processing_purpose=ProcessingPurpose.SENTIMENT_ANALYSIS,
            data_categories={
                DataCategory.COMMUNICATION_CONTENT,
                DataCategory.SENTIMENT_DATA,
                DataCategory.BEHAVIORAL_DATA
            },
            data_processor=processor
        )
        
        # Check if data should be anonymized before processing
        anonymized_data = text_data
        if self.should_anonymize_for_processing(subject_id):
            anonymized_data = self._anonymize_text(text_data)
            
        result = {
            "processing_record_id": record_id,
            "subject_id": subject_id,
            "data_anonymized": anonymized_data != text_data,
            "legal_basis": self.data_subjects[subject_id].legal_basis.value,
            "retention_until": self.processing_records[record_id].retention_until.isoformat() if self.processing_records[record_id].retention_until else None
        }
        
        return result
        
    def handle_data_subject_request(
        self,
        original_id: str,
        request_type: str,
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Handle data subject rights requests (GDPR Articles 15-22).
        
        Args:
            original_id: Original identifier of data subject
            request_type: Type of request (access, rectification, erasure, portability, etc.)
            additional_data: Additional data for the request
            
        Returns:
            Response to the data subject request
        """
        # Find subject by original ID
        subject_id = self._find_subject_by_original_id(original_id)
        
        if not subject_id:
            return {
                "status": "not_found",
                "message": "No data found for the provided identifier"
            }
            
        if request_type == "access":  # Right of access (Article 15)
            return self._handle_access_request(subject_id)
        elif request_type == "rectification":  # Right to rectification (Article 16)
            return self._handle_rectification_request(subject_id, additional_data)
        elif request_type == "erasure":  # Right to erasure (Article 17)
            return self._handle_erasure_request(subject_id)
        elif request_type == "portability":  # Right to data portability (Article 20)
            return self._handle_portability_request(subject_id)
        elif request_type == "objection":  # Right to object (Article 21)
            return self._handle_objection_request(subject_id)
        else:
            return {
                "status": "unsupported",
                "message": f"Request type '{request_type}' is not supported"
            }
            
    def _handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right of access request."""
        data_subject = self.data_subjects[subject_id]
        
        # Find all processing records for this subject
        subject_records = [
            record for record in self.processing_records.values()
            if record.subject_id == subject_id
        ]
        
        processing_activities = []
        for record in subject_records:
            processing_activities.append({
                "purpose": record.processing_purpose.value,
                "categories": [cat.value for cat in record.data_categories],
                "legal_basis": record.legal_basis.value,
                "timestamp": record.timestamp.isoformat(),
                "retention_until": record.retention_until.isoformat() if record.retention_until else None
            })
            
        return {
            "status": "success",
            "data": {
                "subject_id": subject_id,
                "data_categories": [cat.value for cat in data_subject.data_categories],
                "legal_basis": data_subject.legal_basis.value,
                "consent_status": data_subject.consent_given,
                "consent_timestamp": data_subject.consent_timestamp.isoformat() if data_subject.consent_timestamp else None,
                "created_at": data_subject.created_at.isoformat(),
                "processing_activities": processing_activities
            }
        }
        
    def _handle_erasure_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to erasure (right to be forgotten) request."""
        if subject_id not in self.data_subjects:
            return {"status": "not_found"}
            
        data_subject = self.data_subjects[subject_id]
        
        # Check if we have legal grounds to refuse erasure
        refuse_reasons = []
        
        # Check if processing is necessary for compliance with legal obligations
        legal_obligation_records = [
            r for r in self.processing_records.values()
            if r.subject_id == subject_id and r.legal_basis == LegalBasis.LEGAL_OBLIGATION
        ]
        
        if legal_obligation_records:
            refuse_reasons.append("Processing necessary for compliance with legal obligation")
            
        # Check if processing is for public interest/scientific research
        research_records = [
            r for r in self.processing_records.values()
            if r.subject_id == subject_id and r.processing_purpose == ProcessingPurpose.RESEARCH_ANALYSIS
        ]
        
        if research_records:
            refuse_reasons.append("Processing necessary for scientific research purposes")
            
        if refuse_reasons:
            return {
                "status": "refused",
                "reasons": refuse_reasons,
                "message": "Erasure request cannot be fulfilled due to legal grounds for continued processing"
            }
            
        # Proceed with erasure
        erasure_timestamp = datetime.utcnow()
        
        # Anonymize the data instead of complete deletion for audit purposes
        anonymized_id = self._generate_anonymized_id()
        self.anonymization_map[subject_id] = anonymized_id
        
        # Remove from active storage but keep anonymized processing records
        del self.data_subjects[subject_id]
        
        # Anonymize processing records
        for record in self.processing_records.values():
            if record.subject_id == subject_id:
                record.subject_id = anonymized_id
                record.anonymized = True
                
        logger.info(f"Processed erasure request for subject {subject_id}")
        
        return {
            "status": "completed",
            "timestamp": erasure_timestamp.isoformat(),
            "message": "Personal data has been erased/anonymized"
        }
        
    def _handle_portability_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Get structured data for the subject
        access_data = self._handle_access_request(subject_id)
        
        if access_data["status"] != "success":
            return access_data
            
        # Format data for portability (machine-readable format)
        portable_data = {
            "data_export": {
                "format": "json",
                "version": "1.0",
                "exported_at": datetime.utcnow().isoformat(),
                "data": access_data["data"]
            }
        }
        
        return {
            "status": "success",
            "export_data": portable_data,
            "format": "json"
        }
        
    def _handle_objection_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle objection to processing request."""
        if subject_id not in self.data_subjects:
            return {"status": "not_found"}
            
        data_subject = self.data_subjects[subject_id]
        
        # Check if processing is based on legitimate interests
        if data_subject.legal_basis != LegalBasis.LEGITIMATE_INTEREST:
            return {
                "status": "not_applicable",
                "message": "Right to object only applies to processing based on legitimate interests"
            }
            
        # Stop processing based on legitimate interests
        # Mark all future processing as requiring explicit consent
        data_subject.legal_basis = LegalBasis.CONSENT
        data_subject.consent_given = False
        data_subject.consent_timestamp = None
        
        return {
            "status": "accepted",
            "message": "Processing based on legitimate interests has been stopped. Explicit consent will be required for future processing."
        }
        
    def _handle_rectification_request(self, subject_id: str, data_updates: Optional[Dict]) -> Dict[str, Any]:
        """Handle data rectification request."""
        if not data_updates:
            return {
                "status": "invalid", 
                "message": "No rectification data provided"
            }
            
        # For sentiment analysis, rectification might involve updating consent status
        # or data categories, but not historical analysis results
        updated_fields = []
        
        if "consent_status" in data_updates:
            data_subject = self.data_subjects[subject_id]
            data_subject.consent_given = data_updates["consent_status"]
            data_subject.consent_timestamp = datetime.utcnow()
            updated_fields.append("consent_status")
            
        return {
            "status": "completed",
            "updated_fields": updated_fields,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _pseudonymize_id(self, original_id: str) -> str:
        """Create pseudonymized identifier."""
        # Use consistent hashing for pseudonymization
        hash_input = f"{original_id}:{self.data_controller_name}:pseudonym"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def _encrypt_id(self, original_id: str) -> str:
        """Encrypt original identifier for secure storage."""
        # In production, use proper encryption (AES, etc.)
        # This is a simplified example
        hash_input = f"{original_id}:{self.data_controller_name}:encrypted"
        return hashlib.sha256(hash_input.encode()).hexdigest()
        
    def _find_subject_by_original_id(self, original_id: str) -> Optional[str]:
        """Find subject ID by original identifier."""
        encrypted_id = self._encrypt_id(original_id)
        
        for subject_id, data_subject in self.data_subjects.items():
            if data_subject.real_id == encrypted_id:
                return subject_id
                
        return None
        
    def _generate_anonymized_id(self) -> str:
        """Generate anonymous identifier for erasure."""
        return f"anon_{uuid.uuid4().hex[:12]}"
        
    def _anonymize_text(self, text: str) -> str:
        """Anonymize text content by removing personal identifiers."""
        # Simple anonymization - in production use more sophisticated methods
        import re
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove potential names (simple heuristic)
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        
        return text
        
    def should_anonymize_for_processing(self, subject_id: str) -> bool:
        """Determine if data should be anonymized before processing."""
        if not self.auto_anonymize:
            return False
            
        data_subject = self.data_subjects.get(subject_id)
        if not data_subject:
            return True  # Anonymize unknown subjects
            
        # Anonymize if consent is not given and legal basis is not strong enough
        weak_legal_bases = {LegalBasis.LEGITIMATE_INTEREST}
        
        if (not data_subject.consent_given and 
            data_subject.legal_basis in weak_legal_bases):
            return True
            
        return False
        
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up data that has exceeded retention periods."""
        current_time = datetime.utcnow()
        cleaned_subjects = 0
        cleaned_records = 0
        
        # Find expired processing records
        expired_records = []
        for record_id, record in self.processing_records.items():
            if (record.retention_until and 
                current_time > record.retention_until and 
                not record.anonymized):
                expired_records.append(record_id)
                
        # Anonymize expired records
        for record_id in expired_records:
            record = self.processing_records[record_id]
            
            # Check if we should anonymize the subject too
            subject_id = record.subject_id
            if subject_id in self.data_subjects:
                # Check if this is the last record for the subject
                other_active_records = [
                    r for r in self.processing_records.values()
                    if (r.subject_id == subject_id and 
                        r.record_id != record_id and
                        not r.anonymized and
                        (not r.retention_until or current_time <= r.retention_until))
                ]
                
                if not other_active_records:
                    # No other active records, anonymize subject
                    anonymized_id = self._generate_anonymized_id()
                    self.anonymization_map[subject_id] = anonymized_id
                    del self.data_subjects[subject_id]
                    cleaned_subjects += 1
                    
                    # Update all records for this subject
                    for r in self.processing_records.values():
                        if r.subject_id == subject_id:
                            r.subject_id = anonymized_id
                            r.anonymized = True
                            
            cleaned_records += 1
            
        logger.info(f"Cleaned up {cleaned_subjects} subjects and {cleaned_records} expired records")
        
        return {
            "subjects_cleaned": cleaned_subjects,
            "records_cleaned": cleaned_records
        }
        
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for auditing."""
        current_time = datetime.utcnow()
        
        # Count subjects by legal basis
        legal_basis_counts = {}
        for subject in self.data_subjects.values():
            basis = subject.legal_basis.value
            legal_basis_counts[basis] = legal_basis_counts.get(basis, 0) + 1
            
        # Count processing activities by purpose
        purpose_counts = {}
        for record in self.processing_records.values():
            purpose = record.processing_purpose.value
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
            
        # Find upcoming expirations
        upcoming_expirations = 0
        next_30_days = current_time + timedelta(days=30)
        
        for record in self.processing_records.values():
            if (record.retention_until and 
                current_time < record.retention_until < next_30_days):
                upcoming_expirations += 1
                
        report = {
            "report_generated": current_time.isoformat(),
            "data_controller": self.data_controller_name,
            "summary": {
                "total_data_subjects": len(self.data_subjects),
                "total_processing_records": len(self.processing_records),
                "anonymized_subjects": len(self.anonymization_map)
            },
            "legal_basis_distribution": legal_basis_counts,
            "processing_purpose_distribution": purpose_counts,
            "retention_management": {
                "upcoming_expirations_30_days": upcoming_expirations,
                "auto_anonymize_enabled": self.auto_anonymize
            }
        }
        
        return report


# Global instance
_privacy_controller = None


def get_privacy_controller(**kwargs) -> PrivacyController:
    """Get global privacy controller instance."""
    global _privacy_controller
    if _privacy_controller is None:
        _privacy_controller = PrivacyController(**kwargs)
    return _privacy_controller