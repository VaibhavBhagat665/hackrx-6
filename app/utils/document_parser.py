import PyPDF2
import docx
from io import BytesIO
from typing import List, Dict, Tuple
from app.core.logging import logger

class DocumentParser:
    """advanced document parsing with metadata extraction"""
    
    @staticmethod
    def parse_pdf(content: bytes) -> Tuple[str, Dict]:
        """extract text and metadata from pdf"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            
            # extract metadata
            metadata = {
                'pages': len(pdf_reader.pages),
                'doc_type': 'pdf',
                'has_encryption': pdf_reader.is_encrypted
            }
            
            # extract text page by page
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })
            
            full_text = '\n\n'.join([item['text'] for item in text_content])
            metadata['text_pages'] = text_content
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"pdf parsing error: {str(e)}")
            raise ValueError(f"failed to parse pdf: {str(e)}")
    
    @staticmethod
    def parse_docx(content: bytes) -> Tuple[str, Dict]:
        """extract text and metadata from docx"""
        try:
            doc = docx.Document(BytesIO(content))
            
            # extract paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            full_text = '\n\n'.join(paragraphs)
            
            metadata = {
                'paragraphs': len(paragraphs),
                'doc_type': 'docx',
                'word_count': len(full_text.split())
            }
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"docx parsing error: {str(e)}")
            raise ValueError(f"failed to parse docx: {str(e)}")
