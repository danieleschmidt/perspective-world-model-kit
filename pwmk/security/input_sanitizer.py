"""Input sanitization and validation for security."""

import re
from typing import List, Dict, Any, Optional
from ..utils.logging import get_logger


class SecurityError(Exception):
    """Security-related validation error."""
    pass


class InputSanitizer:
    """Sanitizes and validates user inputs for security."""
    
    # Potentially dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'__import__',     # Python imports
        r'eval\s*\(',      # Eval calls
        r'exec\s*\(',      # Exec calls  
        r'open\s*\(',      # File operations
        r'subprocess',     # Process execution
        r'os\.',           # OS operations
        r'sys\.',          # System operations
        r'\.\./',          # Directory traversal
        r'<script',        # XSS attempts
        r'javascript:',    # JavaScript execution
        r'DROP\s+TABLE',   # SQL injection (case insensitive)
        r'DELETE\s+FROM',  # SQL deletion
        r'INSERT\s+INTO',  # SQL insertion
        r'UPDATE\s+SET',   # SQL updates
    ]
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
            for pattern in self.DANGEROUS_PATTERNS
        ]
    
    def sanitize_belief_query(self, query: str) -> str:
        """Sanitize a belief query string."""
        if not isinstance(query, str):
            raise SecurityError("Query must be a string")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(query):
                self.logger.warning(f"Blocked dangerous pattern in query: {query[:100]}")
                raise SecurityError("Query contains potentially dangerous content")
        
        # Limit query length
        if len(query) > 10000:
            raise SecurityError("Query too long (max 10000 characters)")
        
        # Basic sanitization - remove control characters
        sanitized = ''.join(char for char in query if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized
    
    def sanitize_agent_id(self, agent_id: str) -> str:
        """Sanitize agent ID."""
        if not isinstance(agent_id, str):
            raise SecurityError("Agent ID must be a string")
        
        # Remove dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', agent_id)
        
        # Limit length
        if len(sanitized) > 100:
            raise SecurityError("Agent ID too long (max 100 characters)")
        
        if not sanitized:
            raise SecurityError("Agent ID cannot be empty after sanitization")
        
        return sanitized
    
    def sanitize_belief_content(self, belief: str) -> str:
        """Sanitize belief content."""
        if not isinstance(belief, str):
            raise SecurityError("Belief must be a string")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(belief):
                self.logger.warning(f"Blocked dangerous pattern in belief: {belief[:100]}")
                raise SecurityError("Belief contains potentially dangerous content")
        
        # Limit belief length
        if len(belief) > 5000:
            raise SecurityError("Belief too long (max 5000 characters)")
        
        # Sanitize but preserve structure
        sanitized = ''.join(char for char in belief if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized
    
    def validate_file_path(self, path: str) -> bool:
        """Validate file path for security."""
        if not isinstance(path, str):
            return False
        
        # Block directory traversal attempts
        if '..' in path or path.startswith('/'):
            return False
        
        # Only allow safe characters
        if not re.match(r'^[a-zA-Z0-9._/-]+$', path):
            return False
        
        return True


# Global sanitizer instance
_sanitizer = None

def get_sanitizer() -> InputSanitizer:
    """Get global input sanitizer instance."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer()
    return _sanitizer