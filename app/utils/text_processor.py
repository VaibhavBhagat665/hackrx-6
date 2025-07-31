import re
from typing import List, Dict
from app.core.config import settings

class TextProcessor:
    """advanced text processing and chunking"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """clean and normalize text"""
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        # normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    @staticmethod
    def smart_chunk(text: str) -> List[Dict]:
        """intelligent text chunking with semantic boundaries"""
        
        # clean text first
        text = TextProcessor.clean_text(text)
        
        # split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_length = len(words)
            
            # check if adding sentence exceeds chunk size
            if current_length + sentence_length > settings.chunk_size and current_chunk:
                # create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': f"chunk_{len(chunks)}",
                    'word_count': current_length,
                    'sentence_start': i - len(current_chunk),
                    'sentence_end': i - 1
                })
                
                # start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_id': f"chunk_{len(chunks)}",
                'word_count': current_length,
                'sentence_start': len(sentences) - len(current_chunk),
                'sentence_end': len(sentences) - 1
            })
        
        return chunks
