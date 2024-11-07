import re
import unicodedata
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import spacy
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Initialize NLTK components
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: NLTK initialization failed: {e}")

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Warning: spaCy model loading failed: {e}")
    nlp = None

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.common_meeting_terms = {
            'meeting', 'discuss', 'agenda', 'action', 'item', 'decision',
            'follow', 'update', 'review', 'schedule', 'project'
        }
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        return text

    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR and transcription errors."""
        common_fixes = {
            r'\bl\b': 'i',  # Single 'l' to 'i'
            r'\bQ\b': 'O',  # Single 'Q' to 'O'
            r'\b0\b': 'O',  # Single '0' to 'O'
            r'\b1\b': 'I',  # Single '1' to 'I'
            r'[\u201C\u201D]': '"',  # Smart quotes to regular quotes
            r'[\u2018\u2019]': "'",  # Smart apostrophes to regular apostrophes
        }
        
        for pattern, replacement in common_fixes.items():
            text = re.sub(pattern, replacement, text)
            
        return text

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text while maintaining context.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Pre-process to handle special cases
        text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1\n\2', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Post-process sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Fix common sentence boundary errors
                if sentence[-1] not in '.!?':
                    sentence += '.'
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences

    def extract_key_phrases(
        self,
        text: str,
        max_phrases: int = 10,
        min_word_length: int = 2
    ) -> List[Tuple[str, float]]:
        """
        Extract key phrases from text with their importance scores.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to return
            min_word_length: Minimum word length to consider
            
        Returns:
            List of tuples (phrase, score)
        """
        if not nlp:
            return []
            
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            phrase = ' '.join([token.text for token in chunk 
                             if len(token.text) >= min_word_length])
            if phrase:
                noun_phrases.append(phrase)
                
        # Score phrases based on frequency and position
        phrase_scores = {}
        total_phrases = len(noun_phrases)
        
        for idx, phrase in enumerate(noun_phrases):
            # Position score (phrases earlier in text get higher scores)
            position_score = 1 - (idx / total_phrases)
            
            # Length score (prefer medium-length phrases)
            length_score = min(len(phrase.split()), 4) / 4
            
            # Frequency score
            frequency_score = noun_phrases.count(phrase) / total_phrases
            
            # Calculate final score
            final_score = (position_score + length_score + frequency_score) / 3
            phrase_scores[phrase] = final_score
            
        # Sort and return top phrases
        sorted_phrases = sorted(phrase_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        return sorted_phrases[:max_phrases]

    def extract_named_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to sets of entities
        """
        if not nlp:
            return {}
            
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
            
        return entities

    def extract_temporal_expressions(self, text: str) -> List[Dict[str, str]]:
        """
        Extract and normalize temporal expressions from text.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries containing temporal information
        """
        # Regular expressions for common date/time patterns
        patterns = {
            'date': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            'time': r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\b',
            'duration': r'\b(\d+)\s*(hour|minute|second)s?\b',
            'relative': r'\b(tomorrow|yesterday|next|last)\s*(week|month|year)\b'
        }
        
        temporal_expressions = []
        
        for expr_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expression = {
                    'type': expr_type,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                }
                
                # Normalize the expression
                if expr_type == 'date':
                    try:
                        normalized_date = self._normalize_date(match.group(1))
                        expression['normalized'] = normalized_date
                    except ValueError:
                        continue
                        
                temporal_expressions.append(expression)
                
        return sorted(temporal_expressions, key=lambda x: x['start'])

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to ISO format."""
        # Try common date formats
        formats = [
            '%m/%d/%Y', '%m-%d-%Y',
            '%d/%m/%Y', '%d-%m-%Y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%m/%d/%y', '%m-%d-%y'
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        raise ValueError(f"Could not parse date: {date_str}")

    def analyze_text_structure(self, text: str) -> Dict[str, any]:
        """
        Analyze the structure and complexity of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing analysis metrics
        """
        # Basic text metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Calculate readability metrics
        blob = TextBlob(text)
        
        analysis = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_density': len(set(words)) / len(words) if words else 0,
            'sentiment': {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        }
        
        # Add readability scores
        if len(sentences) > 0:
            analysis['readability'] = {
                'flesch_reading_ease': self._calculate_flesch_reading_ease(text),
                'complexity_score': self._calculate_complexity_score(words, sentences)
            }
            
        return analysis

    def _calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        total_words = len(word_tokenize(text))
        total_sentences = len(sent_tokenize(text))
        total_syllables = sum(self._count_syllables(word) for word in text.split())
        
        if total_words == 0 or total_sentences == 0:
            return 0.0
            
        return 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

    def _count_syllables(self, word: str) -> int:
        """Count the number of syllables in a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_char_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_is_vowel:
                count += 1
            prev_char_is_vowel = is_vowel
            
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count

    def _calculate_complexity_score(
        self,
        words: List[str],
        sentences: List[str]
    ) -> float:
        """Calculate text complexity score based on various metrics."""
        if not words or not sentences:
            return 0.0
            
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Unique words ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Complex words (words with 3+ syllables)
        complex_words = sum(1 for word in words 
                          if self._count_syllables(word) >= 3)
        complex_ratio = complex_words / len(words)
        
        # Combine metrics
        complexity_score = (avg_word_length * 0.3 + 
                          unique_ratio * 0.3 + 
                          complex_ratio * 0.4)
        
        return round(complexity_score, 2)

    def segment_text(
        self,
        text: str,
        max_segment_length: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """
        Segment text into overlapping chunks while preserving sentence boundaries.
        
        Args:
            text: Input text
            max_segment_length: Maximum length of each segment
            overlap: Number of characters to overlap between segments
            
        Returns:
            List of text segments
        """
        sentences = self.extract_sentences(text)
        segments = []
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_segment_length and current_segment:
                # Add current segment to results
                segments.append(' '.join(current_segment))
                
                # Start new segment with overlap
                overlap_text = ' '.join(current_segment[-2:])  # Keep last 2 sentences
                current_segment = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current_segment.append(sentence)
                current_length += sentence_length
        
        if current_segment:
            segments.append(' '.join(current_segment))
            
        return segments