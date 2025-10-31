import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extracts structured data from documents based on their type."""

    def __init__(self):
        """Initialize the extractor with regex patterns."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns used for extraction."""
        self.invoice_number_patterns = [
            r'invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'inv\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'invoice\s*number\s*:?\s*([A-Z0-9\-]+)',
        ]

        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  
            r'\d{2}/\d{2}/\d{4}',  
            r'\d{2}-\d{2}-\d{4}',  
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
        ]

        self.amount_patterns = [
            r'total\s*amount\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            r'amount\s*due\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            r'total\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            r'grand\s*total\s*:?\s*\$?\s*([\d,]+\.?\d*)',
        ]

        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        self.phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}-\d{3}-\d{4}',
            r'\(\d{3}\)\s*\d{3}-\d{4}',
        ]

        self.account_number_pattern = r'account\s*number\s*:?\s*([A-Z0-9\-]+)'

        self.usage_pattern = r'(\d+\.?\d*)\s*kwh'

        self.experience_patterns = [
            r'(\d+)\s*\+?\s*years?\s*(?:of)?\s*experience',
            r'experience\s*:?\s*(\d+)\s*\+?\s*years?',
        ]

    def extract(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured data based on document type.

        Args:
            text: Document text
            doc_type: Document classification (Invoice, Resume, etc.)

        Returns:
            Dictionary with extracted fields
        """
        if doc_type == "Invoice":
            return self._extract_invoice(text)
        elif doc_type == "Resume":
            return self._extract_resume(text)
        elif doc_type == "Utility Bill":
            return self._extract_utility_bill(text)
        else:
            return {}

    def extract_batch(self, documents: Dict[str, str], classifications: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract data from multiple documents.
        Args:
            documents: Dictionary mapping filenames to text
            classifications: Dictionary mapping filenames to document types
        Returns:
            Dictionary mapping filenames to extracted data with classification
        """
        results = {}

        for filename, text in documents.items():
            doc_type = classifications.get(filename, "Unclassifiable")
            extracted_data = self.extract(text, doc_type)

            results[filename] = {
                "class": doc_type,
                **extracted_data
            }

        return results

    def _extract_invoice(self, text: str) -> Dict[str, Any]:
        """Extract invoice-specific fields."""
        data = {}

        invoice_number = self._extract_with_patterns(text, self.invoice_number_patterns)
        if invoice_number:
            data['invoice_number'] = invoice_number.upper()

        date = self._extract_date(text)
        if date:
            data['date'] = date

        company = self._extract_company_name(text)
        if company:
            data['company'] = company

        amount = self._extract_amount(text)
        if amount:
            data['total_amount'] = amount

        return data

    def _extract_resume(self, text: str) -> Dict[str, Any]:
        """Extract resume-specific fields."""
        data = {}

        name = self._extract_name(text)
        if name:
            data['name'] = name

        email = self._extract_email(text)
        if email:
            data['email'] = email

        phone = self._extract_phone(text)
        if phone:
            data['phone'] = phone

        experience = self._extract_experience_years(text)
        if experience:
            data['experience_years'] = experience

        return data

    def _extract_utility_bill(self, text: str) -> Dict[str, Any]:
        """Extract utility bill-specific fields."""
        data = {}

        account = self._extract_with_patterns(text, [self.account_number_pattern])
        if account:
            data['account_number'] = account.upper()

        date = self._extract_date(text)
        if date:
            data['date'] = date

        usage = self._extract_usage(text)
        if usage:
            data['usage_kwh'] = usage

        amount = self._extract_amount(text)
        if amount:
            data['amount_due'] = amount

        return data

    def _extract_with_patterns(self, text: str, patterns: list) -> Optional[str]:
        """Extract text using a list of regex patterns."""
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract and normalize dates."""
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(0)
                normalized = self._normalize_date(date_str)
                if normalized:
                    return normalized
        return None

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize various date formats to YYYY-MM-DD."""
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%B %d %Y',
            '%b %d %Y',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract monetary amounts."""
        text_lower = text.lower()
        for pattern in self.amount_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None

    def _extract_company_name(self, text: str) -> Optional[str]:
        """Extract company name using heuristics."""
        # Look for company suffixes
        company_pattern = r'([A-Z][A-Za-z\s&]+(?:Inc|LLC|Ltd|Corporation|Corp|Company|Co)\.?)'
        match = re.search(company_pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_name(self, text: str) -> Optional[str]:
        """Extract person name (heuristic: first line with 2-3 capitalized words)."""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            words = line.strip().split()
            if 2 <= len(words) <= 4:
                # Check if all words are capitalized
                if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                    # Avoid common headers
                    if not any(header in line.lower() for header in ['resume', 'cv', 'curriculum']):
                        return line.strip()
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address."""
        match = re.search(self.email_pattern, text)
        if match:
            return match.group(0)
        return None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        for pattern in self.phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def _extract_experience_years(self, text: str) -> Optional[int]:
        """Extract years of experience."""
        text_lower = text.lower()
        for pattern in self.experience_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def _extract_usage(self, text: str) -> Optional[float]:
        """Extract kWh usage from utility bills."""
        text_lower = text.lower()
        match = re.search(self.usage_pattern, text_lower)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None


extractor = DataExtractor()
