import logging
from typing import Dict, List
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classifies documents using zero-shot classification."""

    DOCUMENT_TYPES = [
        "Invoice",
        "Resume",
        "Utility Bill",
        "Other",
        "Unclassifiable"
    ]

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the classifier with a zero-shot model.

        Args:
            model_name: Hugging Face model to use for classification
        """
        logger.info(f"Loading classification model: {model_name}")

        device = 0 if torch.cuda.is_available() else -1

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )

        self.hypothesis_template = "This document is a {}."

        logger.info("Classifier initialized successfully")

    def classify(self, text: str, filename: str = "") -> str:
        """
        Classify a document into one of the predefined categories.

        Args:
            text: Document text to classify
            filename: Optional filename for additional context

        Returns:
            Document type (Invoice, Resume, Utility Bill, Other, or Unclassifiable)
        """
        if not text or len(text.strip()) < 10:
            logger.warning(f"Insufficient text for classification: {filename}")
            return "Unclassifiable"

        try:
            text_sample = text[:2000]

            result = self.classifier(
                text_sample,
                candidate_labels=self.DOCUMENT_TYPES,
                hypothesis_template=self.hypothesis_template,
                multi_label=False
            )

            predicted_class = result['labels'][0]
            confidence = result['scores'][0]

            logger.info(f"{filename}: {predicted_class} (confidence: {confidence:.2f})")

            if confidence < 0.3:
                return "Unclassifiable"

            predicted_class = self._apply_rules(text, predicted_class, confidence)

            return predicted_class

        except Exception as e:
            logger.error(f"Classification failed for {filename}: {e}")
            return "Unclassifiable"

    def classify_batch(self, documents: Dict[str, str]) -> Dict[str, str]:
        """
        Classify multiple documents.

        Args:
            documents: Dictionary mapping filenames to document text

        Returns:
            Dictionary mapping filenames to document types
        """
        classifications = {}

        for filename, text in documents.items():
            doc_type = self.classify(text, filename)
            classifications[filename] = doc_type

        return classifications

    def _apply_rules(self, text: str, predicted_class: str, confidence: float) -> str:
        """
        Apply rule-based refinements to improve classification accuracy.

        Args:
            text: Document text
            predicted_class: Predicted class from the model
            confidence: Confidence score

        Returns:
            Refined document class
        """
        text_lower = text.lower()

        invoice_keywords = [
            'invoice', 'invoice number', 'invoice #', 'bill to',
            'total amount', 'amount due', 'payment terms', 'subtotal',
            'tax', 'vat', 'due date'
        ]

        resume_keywords = [
            'resume', 'curriculum vitae', 'cv', 'experience',
            'education', 'skills', 'objective', 'professional summary',
            'work history', 'employment', 'qualifications'
        ]

        utility_keywords = [
            'utility', 'electric', 'electricity', 'gas', 'water',
            'kwh', 'kilowatt', 'meter', 'usage', 'service address',
            'account number', 'billing period', 'current charges'
        ]

        invoice_count = sum(1 for kw in invoice_keywords if kw in text_lower)
        resume_count = sum(1 for kw in resume_keywords if kw in text_lower)
        utility_count = sum(1 for kw in utility_keywords if kw in text_lower)

        if invoice_count >= 3 and predicted_class != "Invoice":
            logger.info(f"Rule override: Changing to Invoice (found {invoice_count} keywords)")
            return "Invoice"

        if resume_count >= 3 and predicted_class != "Resume":
            logger.info(f"Rule override: Changing to Resume (found {resume_count} keywords)")
            return "Resume"

        if utility_count >= 3 and predicted_class != "Utility Bill":
            logger.info(f"Rule override: Changing to Utility Bill (found {utility_count} keywords)")
            return "Utility Bill"

        if confidence < 0.5 and max(invoice_count, resume_count, utility_count) < 2:
            return "Other"

        return predicted_class


classifier = DocumentClassifier()

