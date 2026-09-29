"""TABULA OCR: schema-locked document extraction with evidence and abstention.

Public surface:

>>> from tabula_ocr import OCRService, Settings
"""

from tabula_ocr.config import Settings, get_settings
from tabula_ocr.models import DocumentResult, FieldResult, FieldStatus
from tabula_ocr.pipeline import OCRService

__version__ = "1.0.0"

__all__ = [
    "DocumentResult",
    "FieldResult",
    "FieldStatus",
    "OCRService",
    "Settings",
    "__version__",
    "get_settings",
]
