"""
CEFR Grammar Checker - Python Scoring Microservice
Advanced 5-category rubric scoring with linguistic analysis
"""

import sys
import os

# Check Python version first
print(f"=== Starting CEFR API ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import spacy  # Disabled for free hosting
import re
import json
import hashlib
import hmac
import time
from collections import Counter
import math
import nltk
from cefrpy import CEFRAnalyzer  # Commented out - not available on free hosting
from nltk.corpus import wordnet as wn

app = FastAPI(title="CEFR Scoring Service", version="1.0.0")

# Configure CORS for secure frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://*.replit.app",
        "https://*.replit.dev",
        "https://darkslatechrome-sparrow-221218.hostingersite.com",
        "http://darkslatechrome-sparrow-221218.hostingersite.com",
        "https://papayawhip-elk-540494.hostingersite.com/wp-admin/admin.php?page=hostinger",
        "https://papayawhip-elk-540494.hostingersite.com",
        "http://papayawhip-elk-540494.hostingersite.com"
    ],
    allow_credentials=False,  # Disabled for security
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# Initialize language models and tools  
nlp = None  # Disabled for free hosting compatibility

# Initialize NLTK and download required data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"NLTK download error: {e}")

# Initialize CEFR analyzer
try:
    # cefr_analyzer = CEFRAnalyzer()  # Commented out - not available on free hosting
    # print("CEFR analyzer initialized successfully")
    cefr_analyzer = None  # Use simplified analysis for now
    print("CEFR analyzer disabled for free hosting compatibility")
except Exception as e:
    print(f"CEFR analyzer initialization error: {e}")
    cefr_analyzer = None

class TextAnalysisRequest(BaseModel):
    text: str
    task_type: Optional[str] = "general"
    min_words: Optional[int] = 10
    max_words: Optional[int] = 1000
    language: Optional[str] = "en"

class RubricScore(BaseModel):
    vocabulary_range_appropriacy: float
    grammar_range_accuracy: float
    coherence_cohesion: float
    task_fulfillment_register: float
    mechanics_punctuation: float
    total_score: float
    cefr_level: str
    cefr_sublevel: str

class AnalysisEvidence(BaseModel):
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    lexical_diversity: float
    advanced_vocabulary_ratio: float
    grammar_errors: List[Dict]
    coherence_markers: List[str]
    punctuation_errors: List[Dict]
    register_analysis: Dict
    semantic_similarity: Optional[float] = 0.0
    error_rate: Optional[float] = 0.0

class WordAnalysisRequest(BaseModel):
    word: str
    include_synonyms: Optional[bool] = True
    include_examples: Optional[bool] = True

class WordAnalysisResponse(BaseModel):
    word: str
    cefr_level: Optional[str]
    definitions: List[Dict]
    synonyms: List[str]
    antonyms: List[str]
    part_of_speech: List[str]
    examples: List[str]
    pronunciation: Optional[str]
    frequency_score: Optional[float]

class CorrectionSymbol(BaseModel):
    symbol: str
    category: str
    description: str
    error_text: str
    suggestion: str
    position: Optional[int] = None

class CategoryAssessment(BaseModel):
    strengths: List[str]
    range_or_structures: Optional[str] = None
    errors_by_symbol: Dict[str, int] = Field(default_factory=dict)
    score: float
    feedback: List[str]

class AssessmentFeedback(BaseModel):
    vocabulary: CategoryAssessment
    grammar: CategoryAssessment
    coherence_cohesion: CategoryAssessment
    task_fulfillment_register: CategoryAssessment
    mechanics_punctuation: CategoryAssessment
    total_score: float
    cefr_level: str
    cefr_sublevel: str
    error_summary_by_symbol: Dict[str, int]
    correction_symbols: List[CorrectionSymbol]
    narrative_feedback: str

class EnhancedScoringResponse(BaseModel):
    rubric: RubricScore
    evidence: AnalysisEvidence
    feedback: AssessmentFeedback
    text_validity: Dict[str, bool]
    processing_time: float

class ScoringResponse(BaseModel):
    rubric: RubricScore
    evidence: AnalysisEvidence
    processing_time: float

# Comprehensive Correction Symbols System
CORRECTION_SYMBOLS = {
    # Grammar & Usage (20 symbols)
    'agr': {
        'category': 'Grammar & Usage',
        'description': 'Subject/verb agreement error',
        'patterns': [r'\b(I|you|we|they)\s+(?:is|was|has)\b', r'\b(he|she|it)\s+(?:are|were|have)\b']
    },
    'awk': {
        'category': 'Grammar & Usage', 
        'description': 'Awkward phrasing or construction',
        'patterns': [r'\b(?:very|really|so|quite)\s+(?:very|really|so|quite)\b']
    },
    'cs': {
        'category': 'Grammar & Usage',
        'description': 'Comma splice - two independent clauses joined by comma',
        'patterns': [r'\w+,\s+\w+\s+(?:is|are|was|were|has|have|can|will|would)\s']
    },
    'dm': {
        'category': 'Grammar & Usage',
        'description': 'Dangling or misplaced modifier',
        'patterns': [r'^(?:Having|Being|After|Before)\s.*,\s(?:the|a|an)\s']
    },
    'frag': {
        'category': 'Grammar & Usage',
        'description': 'Sentence fragment',
        'patterns': [r'\.\s+(?:Because|Since|Although|While|If)\s[^.]*\.\s']
    },
    '//': {
        'category': 'Grammar & Usage',
        'description': 'Faulty parallelism in structure',
        'patterns': [r'\b(?:and|or)\s+(?:to\s+\w+|\w+ing)\b']
    },
    'ref': {
        'category': 'Grammar & Usage',
        'description': 'Unclear pronoun reference',
        'patterns': [r'\b(?:this|that|it|they)\s+(?:is|are|was|were)\b']
    },
    'ro': {
        'category': 'Grammar & Usage',
        'description': 'Run-on sentence',
        'patterns': [r'\w+\s+(?:and|but|so|or)\s+\w+\s+(?:and|but|so|or)\s+\w+']
    },
    't': {
        'category': 'Grammar & Usage',
        'description': 'Verb tense error',
        'patterns': [r'\b(?:yesterday|last\s+\w+)\s+.*\s+(?:will|is|are)\s']
    },
    'wo': {
        'category': 'Grammar & Usage',
        'description': 'Wrong word order',
        'patterns': [r'\b(?:always|never|often|sometimes)\s+(?:is|are|was|were)\b']
    },
    'ww': {
        'category': 'Grammar & Usage',
        'description': 'Wrong word choice',
        'patterns': [r'\b(?:there|their|they\'re)\b', r'\b(?:your|you\'re)\b', r'\b(?:its|it\'s)\b']
    },
    'prep': {
        'category': 'Grammar & Usage',
        'description': 'Preposition error',
        'patterns': [r'\bon\s+the\s+internet\b', r'\bin\s+monday\b', r'\bat\s+home\s+country\b']
    },
    'art': {
        'category': 'Grammar & Usage',
        'description': 'Article error (a, an, the)',
        'patterns': [r'\ba\s+(?:[aeiou])\w+\b', r'\ban\s+(?:[bcdfghjklmnpqrstvwxyz])\w+\b', r'\bthe\s+(?:happiness|love|music)\b']
    },
    'mod': {
        'category': 'Grammar & Usage',
        'description': 'Misplaced or incorrect modifier',
        'patterns': [r'\b(?:only|just|even|almost|nearly)\s+(?:can|will|should|could|would)\b']
    },
    'sub': {
        'category': 'Grammar & Usage',
        'description': 'Subordination error',
        'patterns': [r'\b(?:because|since|although|while)\s+but\b', r'\balthough.*but\b']
    },
    'coord': {
        'category': 'Grammar & Usage',
        'description': 'Coordination error',
        'patterns': [r'\band\s+but\b', r'\bbut\s+and\b', r'\bor\s+and\b']
    },
    'pass': {
        'category': 'Grammar & Usage',
        'description': 'Inappropriate passive voice',
        'patterns': [r'\b(?:mistakes\s+were\s+made|it\s+is\s+believed\s+that)\b']
    },
    'inf': {
        'category': 'Grammar & Usage',
        'description': 'Infinitive error',
        'patterns': [r'\bto\s+not\s+\w+\b', r'\bfor\s+to\s+\w+\b']
    },
    'ger': {
        'category': 'Grammar & Usage',
        'description': 'Gerund error',
        'patterns': [r'\bappreciate\s+to\s+\w+\b', r'\bavoid\s+to\s+\w+\b']
    },
    'part': {
        'category': 'Grammar & Usage',
        'description': 'Participle error',
        'patterns': [r'\bhaving\s+\w+ed\s+yesterday\b', r'\bbeing\s+that\b']
    },
    
    # Mechanics & Punctuation (15 symbols)
    'sp': {
        'category': 'Mechanics & Punctuation',
        'description': 'Spelling error',
        'patterns': [r'\b(?:recieve|seperate|occured|neccessary|definately|accomodate|occassion|recomend)\b']
    },
    'punc': {
        'category': 'Mechanics & Punctuation',
        'description': 'General punctuation error',
        'patterns': [r'\w+\s+\w+[.!?]\w+', r'\w+[,;:]\w+']
    },
    'apos': {
        'category': 'Mechanics & Punctuation',
        'description': 'Apostrophe error',
        'patterns': [r'\b\w+s\'\s+\w+\b', r'\bit\'s\s+(?:car|book|house)\b']
    },
    'caps': {
        'category': 'Mechanics & Punctuation',
        'description': 'Capitalization needed',
        'patterns': [r'\.\s+[a-z]', r'\bi\s+(?:am|was|will|have)\b']
    },
    '^': {
        'category': 'Mechanics & Punctuation',
        'description': 'Missing element (word, punctuation)',
        'patterns': [r'\bthe\s+\w+\s+of\s+(?!the)\w+\b', r'\ba\s+(?:[aeiou])\w+\b']
    },
    'delete': {
        'category': 'Mechanics & Punctuation',
        'description': 'Remove unnecessary text',
        'patterns': [r'\b(?:very|really|quite|totally)\s+(?:unique|perfect|impossible)\b']
    },
    'tr': {
        'category': 'Mechanics & Punctuation',
        'description': 'Transpose/reorder elements',
        'patterns': [r'\b(?:only|just|even)\s+(?:can|will|should)\b']
    },
    '¶': {
        'category': 'Mechanics & Punctuation',
        'description': 'New paragraph needed',
        'patterns': [r'\.\s+(?:However|Furthermore|Moreover|In\s+contrast)\s']
    },
    '#': {
        'category': 'Mechanics & Punctuation',
        'description': 'Add a space',
        'patterns': [r'\w+\.\w+', r'\w+,\w+', r'\w+;\w+']
    },
    'quot': {
        'category': 'Mechanics & Punctuation',
        'description': 'Quotation marks error',
        'patterns': [r'[.!?]"\s*[a-z]', r'"\s*[.!?]', r"'\s*[.!?]"]
    },
    'semi': {
        'category': 'Mechanics & Punctuation',
        'description': 'Semicolon error',
        'patterns': [r';\s*[A-Z]', r';\s*and\s', r';\s*but\s']
    },
    'colon': {
        'category': 'Mechanics & Punctuation',
        'description': 'Colon error',
        'patterns': [r':\s*[a-z]', r'\w+:\w+']
    },
    'hyph': {
        'category': 'Mechanics & Punctuation',
        'description': 'Hyphenation error',
        'patterns': [r'\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(?:one|two|three|four|five|six|seven|eight|nine)\b']
    },
    'ital': {
        'category': 'Mechanics & Punctuation',
        'description': 'Italics needed for titles, foreign words',
        'patterns': [r'\b(?:et\s+al|i\.e\.|e\.g\.)\b', r'\bHamlet\b', r'\bThe\s+New\s+York\s+Times\b']
    },
    'abbr': {
        'category': 'Mechanics & Punctuation',
        'description': 'Abbreviation error',
        'patterns': [r'\bU\.S\.A\b', r'\betc\b(?!\.)']
    },
    
    # Style & Clarity (12 symbols)
    'cl': {
        'category': 'Style & Clarity',
        'description': 'Lacks clarity - unclear meaning',
        'patterns': [r'\b(?:thing|stuff|it|this)\s+(?:is|are|was|were)\s+(?:good|bad|nice)\b']
    },
    'coh': {
        'category': 'Style & Clarity',
        'description': 'Poor coherence between ideas',
        'patterns': [r'\.\s+(?:And|But|So)\s+[^.]*\.\s+(?:And|But|So)\s']
    },
    'dev': {
        'category': 'Style & Clarity',
        'description': 'Needs development - expand the idea',
        'patterns': [r'\b(?:good|bad|nice|fine|okay)\b\.\s']
    },
    'rep': {
        'category': 'Style & Clarity',
        'description': 'Repetition - avoid repeating words/ideas',
        'patterns': [r'\b(\w+)\s+.*\s+\1\b', r'\b(\w+)\s+\1\b']
    },
    'wordy': {
        'category': 'Style & Clarity',
        'description': 'Wordiness - too many unnecessary words',
        'patterns': [r'\b(?:in\s+order\s+to|due\s+to\s+the\s+fact\s+that|at\s+this\s+point\s+in\s+time)\b']
    },
    'tone': {
        'category': 'Style & Clarity',
        'description': 'Inappropriate tone for context',
        'patterns': [r'\b(?:gonna|wanna|ain\'t|yeah|nope)\b', r'\b(?:awesome|cool|sucks)\b']
    },
    'form': {
        'category': 'Style & Clarity',
        'description': 'Inappropriate formality level',
        'patterns': [r'\b(?:I\s+think|I\s+believe|in\s+my\s+opinion)\b']
    },
    'bias': {
        'category': 'Style & Clarity',
        'description': 'Biased or discriminatory language',
        'patterns': [r'\b(?:fireman|policeman|chairman)\b', r'\b(?:guys)\b.*(?:everyone|all)']
    },
    'shift': {
        'category': 'Style & Clarity',
        'description': 'Unnecessary shift in person, tense, or voice',
        'patterns': [r'\bwe\s+.*you\s+.*\b', r'\bI\s+.*one\s+.*\b']
    },
    'trans': {
        'category': 'Style & Clarity',
        'description': 'Transition needed between ideas',
        'patterns': [r'\.\s+[A-Z]\w+\s+(?:is|are|was|were)\s', r'\.\s+(?!However|Furthermore|Moreover|Therefore)']
    },
    'emp': {
        'category': 'Style & Clarity',
        'description': 'Emphasis needed or inappropriate emphasis',
        'patterns': [r'\b(?:very|really|extremely|incredibly)\s+(?:very|really|extremely|incredibly)\b']
    },
    'var': {
        'category': 'Style & Clarity',
        'description': 'Sentence variety needed',
        'patterns': [r'^[A-Z]\w+\s+(?:is|are|was|were)\s.*\.\s*[A-Z]\w+\s+(?:is|are|was|were)\s.*\.\s*[A-Z]\w+\s+(?:is|are|was|were)\s']
    },
    
    # Vocabulary (7 symbols)
    'col': {
        'category': 'Vocabulary',
        'description': 'Collocation error - wrong word combination',
        'patterns': [r'\b(?:make|do)\s+(?:homework|research|mistake)\b', r'\b(?:strong|heavy)\s+(?:tea|rain)\b']
    },
    'idio': {
        'category': 'Vocabulary',
        'description': 'Non-idiomatic expression',
        'patterns': [r'\blearn\s+by\s+heart\b', r'\bmake\s+sport\b', r'\bsay\s+lie\b']
    },
    'dict': {
        'category': 'Vocabulary',
        'description': 'Poor word choice or diction',
        'patterns': [r'\b(?:utilize|commence|terminate)\b', r'\bget\s+(?:angry|happy|sad)\b']
    },
    'jarg': {
        'category': 'Vocabulary',
        'description': 'Inappropriate jargon for audience',
        'patterns': [r'\b(?:synergy|leverage|paradigm|utilize)\b']
    },
    'slang': {
        'category': 'Vocabulary',
        'description': 'Inappropriate slang for context',
        'patterns': [r'\b(?:cool|awesome|rad|sick|dope|lit)\b']
    },
    'arch': {
        'category': 'Vocabulary',
        'description': 'Archaic or outdated word choice',
        'patterns': [r'\b(?:thou|thee|thy|wherefore|whilst|amongst)\b']
    },
    'neol': {
        'category': 'Vocabulary',
        'description': 'Neologism or non-standard word',
        'patterns': [r'\b(?:impactful|conversate|irregardless|orientate)\b']
    }
}

# CEFR Level Mapping (0-25 points total)
# Based on specification: 0-6→A1, 7-12→A2, 13-17→B1, 18-20→B2, 21-23→C1, 24-25→C2
CEFR_MAPPING = {
    (0, 2): ("A1", "low"),
    (3, 4): ("A1", "mid"), 
    (5, 6): ("A1", "high"),
    (7, 9): ("A2", "low"),
    (10, 11): ("A2", "mid"),
    (12, 12): ("A2", "high"), 
    (13, 14): ("B1", "low"),
    (15, 16): ("B1", "mid"),
    (17, 17): ("B1", "high"),
    (18, 18): ("B2", "low"),
    (19, 19): ("B2", "mid"),
    (20, 20): ("B2", "high"),
    (21, 21): ("C1", "low"),
    (22, 22): ("C1", "mid"),
    (23, 23): ("C1", "high"),
    (24, 24): ("C2", "low"),
    (25, 25): ("C2", "high")
}

def get_cefr_level(total_score: float) -> tuple:
    """Map total score (0-25) to CEFR level and sublevel"""
    # Direct mapping without scaling
    score = int(round(total_score))
    
    for (min_score, max_score), (level, sublevel) in CEFR_MAPPING.items():
        if min_score <= score <= max_score:
            return level, sublevel
    
    # Fallback
    return ("C2", "high") if score > 25 else ("A1", "low")

def check_text_validity(text: str, min_words: int = 50) -> Dict[str, bool]:
    """Enhanced text validity checks"""
    words = text.split()
    word_count = len(words)
    
    # Basic length check
    length_valid = word_count >= min_words
    
    # Check for minimum sentence structure
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    structure_valid = sentence_count >= 3
    
    # Check for basic vocabulary diversity
    unique_words = len(set(words))
    diversity_valid = unique_words / word_count >= 0.4 if word_count > 0 else False
    
    # Check for coherence markers
    coherence_markers = ['however', 'therefore', 'furthermore', 'moreover', 'in addition', 'consequently', 'nevertheless']
    has_coherence = any(marker in text.lower() for marker in coherence_markers)
    
    # Check for appropriate register (not too informal for assessment)
    informal_markers = ["don't", "can't", "won't", "it's", "that's", "i'm"]
    informality_ratio = sum(1 for marker in informal_markers if marker in text.lower()) / word_count if word_count > 0 else 0
    register_appropriate = informality_ratio < 0.1  # Less than 10% informal markers
    
    return {
        'length_adequate': length_valid,
        'structure_adequate': structure_valid,
        'vocabulary_diverse': diversity_valid,
        'has_coherence_markers': has_coherence,
        'register_appropriate': register_appropriate,
        'overall_valid': all([length_valid, structure_valid, diversity_valid, has_coherence, register_appropriate])
    }

def analyze_correction_symbols(text: str) -> List[CorrectionSymbol]:
    """Detect errors and assign correction symbols"""
    symbols = []
    
    for symbol, data in CORRECTION_SYMBOLS.items():
        for pattern in data.get('patterns', []):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                symbols.append(CorrectionSymbol(
                    symbol=symbol,
                    category=data['category'],
                    description=data['description'],
                    error_text=match.group(),
                    suggestion=generate_correction_suggestion(symbol, match.group()),
                    position=match.start()
                ))
    
    return symbols

def generate_correction_suggestion(symbol: str, error_text: str) -> str:
    """Generate specific correction suggestions based on symbol"""
    suggestions = {
        # Grammar & Usage (20 symbols)
        'agr': f"Check subject-verb agreement. Consider: {error_text.replace('is', 'are').replace('are', 'is')}",
        'awk': "Rephrase for clearer, more natural expression",
        'cs': "Avoid comma splices - use semicolon or separate sentences",
        'dm': "Ensure modifiers clearly refer to the intended word",
        'frag': "Complete the sentence with a main clause",
        '//': "Make parallel elements consistent in structure",
        'ref': "Clarify what the pronoun refers to",
        'ro': "Break this into separate sentences or use proper conjunctions",
        't': "Review verb tense consistency throughout the text",
        'wo': "Adjust word order for clarity and correctness",
        'ww': "Consider the correct word choice here",
        'prep': "Use the correct preposition for this context",
        'art': "Check article usage (a, an, the) - consider countability and specificity",
        'mod': "Place modifiers near the words they modify",
        'sub': "Check subordinating conjunctions and clause structure",
        'coord': "Use appropriate coordinating conjunctions",
        'pass': "Consider using active voice for clarity",
        'inf': "Check infinitive form and usage",
        'ger': "Review gerund usage after this verb",
        'part': "Check participle form and placement",
        
        # Mechanics & Punctuation (15 symbols)
        'sp': f"Check spelling of '{error_text}'",
        'punc': f"Add appropriate punctuation around '{error_text}'",
        'apos': "Check apostrophe usage for possession or contractions",
        'caps': f"Capitalize the first letter: {error_text.capitalize()}",
        '^': "Add the missing word or punctuation mark",
        'delete': "Remove unnecessary words or phrases",
        'tr': "Reorder these elements for better flow",
        '¶': "Start a new paragraph here for better organization",
        '#': "Add a space between these elements",
        'quot': "Check quotation mark placement and punctuation",
        'semi': "Use semicolon to connect related independent clauses",
        'colon': "Use colon to introduce lists or explanations",
        'hyph': "Add hyphen to compound words or numbers",
        'ital': "Use italics for titles, foreign words, or emphasis",
        'abbr': "Follow standard abbreviation rules",
        
        # Style & Clarity (12 symbols)
        'cl': "Clarify this expression for better understanding",
        'coh': "Add transitions or reorganize for better flow",
        'dev': "Expand this idea with more detail or examples",
        'rep': "Avoid repetition by using synonyms or restructuring",
        'wordy': "Simplify this phrase for better clarity",
        'tone': "Adjust tone to match your audience and purpose",
        'form': "Match formality level to context and audience",
        'bias': "Use inclusive, non-discriminatory language",
        'shift': "Maintain consistent person, tense, and voice",
        'trans': "Add transition words to connect ideas",
        'emp': "Adjust emphasis for better impact",
        'var': "Vary sentence structure and length",
        
        # Vocabulary (7 symbols)
        'col': "Check the word combination - consider standard collocations",
        'idio': "Use standard idiomatic expressions",
        'dict': "Choose more precise or appropriate vocabulary",
        'jarg': "Replace jargon with clear, accessible language",
        'slang': "Use formal language appropriate for academic writing",
        'arch': "Replace archaic words with modern alternatives",
        'neol': "Use standard, established vocabulary"
    }
    return suggestions.get(symbol, f"Review and correct the error marked with '{symbol}'")

def analyze_enhanced_nlp_features(doc, text: str) -> Dict:
    """Enhanced NLP feature analysis for comprehensive assessment"""
    if not doc:
        return {}
    
    # Optimize: Single pass through doc for multiple analyses
    tokens = []
    advanced_words = []
    alpha_tokens = []
    
    for token in doc:
        if token.is_alpha:
            alpha_tokens.append(token)
            if not token.is_stop:
                tokens.append(token.lemma_.lower())
            if len(token.text) > 7 and token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                advanced_words.append(token)
    
    # Lexical richness analysis
    unique_tokens = set(tokens)
    lexical_diversity = len(unique_tokens) / len(tokens) if tokens else 0
    
    # Advanced vocabulary ratio
    advanced_ratio = len(advanced_words) / len(alpha_tokens) if alpha_tokens else 0
    
    # Sentence complexity (cache sentences list)
    sentences = list(doc.sents)
    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
    complex_sentences = sum(1 for sent in sentences if len(sent) > 15)
    complexity_ratio = complex_sentences / len(sentences) if sentences else 0
    
    # Optimize: Pre-convert text to lowercase once and use compiled patterns
    text_lower = text.lower()
    
    # Coherence markers (optimized with set lookup)
    coherence_words = {'however', 'therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless', 
                      'in addition', 'for example', 'in contrast', 'on the other hand', 'as a result'}
    coherence_count = sum(1 for word in coherence_words if word in text_lower)
    coherence_density = coherence_count / len(sentences) if sentences else 0
    
    # Register consistency analysis (optimized with set lookup)
    formal_markers = {'shall', 'ought', 'furthermore', 'moreover', 'thus', 'hence', 'consequently'}
    informal_markers = {"don't", "can't", "won't", "it's", "that's", "gonna", "wanna"}
    
    formal_count = sum(1 for marker in formal_markers if marker in text_lower)
    informal_count = sum(1 for marker in informal_markers if marker in text_lower)
    
    register_consistency = abs(formal_count - informal_count) / max(formal_count + informal_count, 1)
    
    # Error rate analysis
    correction_symbols = analyze_correction_symbols(text)
    error_rate = len(correction_symbols) / len(tokens) if tokens else 0
    
    # Semantic similarity analysis (simplified implementation)
    sentences = list(doc.sents)
    semantic_similarity = 0.0
    if len(sentences) > 1:
        # Calculate average similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i + 1]
            if sent1.vector.any() and sent2.vector.any():
                similarity = sent1.similarity(sent2)
                similarities.append(similarity)
        semantic_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    return {
        'lexical_diversity': lexical_diversity,
        'advanced_vocabulary_ratio': advanced_ratio,
        'avg_sentence_length': avg_sentence_length,
        'sentence_complexity_ratio': complexity_ratio,
        'coherence_marker_density': coherence_density,
        'register_consistency_score': register_consistency,
        'error_rate': error_rate,
        'correction_symbols': correction_symbols,
        'semantic_similarity': semantic_similarity
    }

def generate_comprehensive_feedback(rubric: RubricScore, features: Dict, correction_symbols: List[CorrectionSymbol], text: str, doc) -> AssessmentFeedback:
    """Generate comprehensive feedback following the exact CEFR assessment format"""
    
    # Count errors by symbol
    error_counts_by_symbol = {}
    for symbol in correction_symbols:
        error_counts_by_symbol[symbol.symbol] = error_counts_by_symbol.get(symbol.symbol, 0) + 1
    
    # 1. Vocabulary Assessment (X/5)
    vocab_strengths = []
    vocab_range = "basic"
    vocab_feedback = []
    vocab_errors = {}
    
    # Analyze vocabulary range and topic-specific terms
    if rubric.vocabulary_range_appropriacy >= 4.0:
        vocab_strengths.append("Effective use of topic-specific terms")
        vocab_range = "varied but not advanced idiomatic"
    elif rubric.vocabulary_range_appropriacy >= 3.0:
        vocab_strengths.append("Good use of varied vocabulary")
        vocab_range = "varied"
    else:
        vocab_range = "basic"
    
    # Check for specific vocabulary features
    advanced_words = len([w for w in text.split() if len(w) > 7])
    if advanced_words > 5:
        vocab_strengths.append("Good contrast markers and connective language")
    
    # Vocabulary errors - focus on word choice (ww), collocations (col), idioms (idio)
    for symbol in ['ww', 'col', 'idio']:
        if symbol in error_counts_by_symbol:
            vocab_errors[symbol] = error_counts_by_symbol[symbol]
    
    # Generate specific vocabulary feedback
    if 'ww' in vocab_errors:
        vocab_feedback.append("Work on word choice accuracy (e.g., verb forms and prepositions)")
    if vocab_errors:
        vocab_feedback.append("Add more precise collocations (e.g., quality time, digital interaction)")
        vocab_feedback.append("Incorporate idiomatic phrases for nuance (e.g., double-edged sword)")
    else:
        vocab_feedback.append("Expand academic vocabulary and sophisticated collocations")
    
    # 2. Grammar Assessment (X/5)
    grammar_strengths = []
    grammar_feedback = []
    grammar_errors = {}
    
    # Analyze grammar structures
    if doc:
        # Check for contrastive structures
        contrastive_markers = ['however', 'on the other hand', 'nevertheless', 'whereas']
        has_contrastive = any(marker in text.lower() for marker in contrastive_markers)
        if has_contrastive:
            grammar_strengths.append("Correct use of contrastive structures (on the other hand)")
    
    # Grammar errors - focus on agreement, tense, word order
    for symbol in ['agr', 'cs', 'frag', 'ro', '//', 'ref', 't', 'wo', 'awk', 'dm']:
        if symbol in error_counts_by_symbol:
            grammar_errors[symbol] = error_counts_by_symbol[symbol]
    
    # Generate specific grammar feedback
    if 'agr' in grammar_errors:
        grammar_feedback.append(f"Work on subject–verb agreement (agr) - found {grammar_errors['agr']} errors")
    if 'ww' in error_counts_by_symbol:
        grammar_feedback.append("Practice article/preposition accuracy")
    if grammar_errors:
        grammar_feedback.append("Practice advanced forms like conditionals (If people relied less on technology, family bonds might improve)")
    else:
        grammar_feedback.append("Continue practicing complex grammatical structures")
    
    # 3. Coherence & Cohesion Assessment (X/5)
    coherence_strengths = []
    coherence_feedback = []
    coherence_errors = {}
    
    # Analyze text structure
    sentences = text.split('.')
    if len(sentences) >= 4:
        coherence_strengths.append("Clear structure: opinion → example → counterpoint → conclusion")
    
    # Check for advanced connectors
    advanced_connectors = ['however', 'furthermore', 'moreover', 'consequently', 'nevertheless', 'in contrast']
    basic_connectors = ['and', 'but', 'so', 'because']
    
    text_lower = text.lower()
    advanced_count = sum(1 for conn in advanced_connectors if conn in text_lower)
    if advanced_count > 0:
        coherence_strengths.append("Good use of discourse markers")
    
    # Coherence errors
    for symbol in ['coh', 'awk', 'rep', 'wordy', 'cl']:
        if symbol in error_counts_by_symbol:
            coherence_errors[symbol] = error_counts_by_symbol[symbol]
    
    # Generate coherence feedback
    if 'awk' in coherence_errors:
        coherence_feedback.append("Minor cohesion issues with awkward flow (awk)")
    coherence_feedback.append("Use advanced connectors (e.g., nevertheless, furthermore, in contrast) to elevate cohesion")
    coherence_feedback.append("Smooth transitions between opinion and counter-argument")
    
    # 4. Task Fulfillment & Register Assessment (X/5)
    task_strengths = []
    task_feedback = []
    task_errors = {}
    
    # Analyze task completion and register
    word_count = len(text.split())
    if word_count >= 150:
        task_strengths.append("Fully addresses the prompt")
        task_strengths.append("Balanced argument with example and conclusion")
    
    # Check register consistency
    formal_markers = ['furthermore', 'however', 'consequently', 'therefore']
    informal_markers = ["don't", "can't", "won't", "it's"]
    
    formal_count = sum(1 for marker in formal_markers if marker in text_lower)
    informal_count = sum(1 for marker in informal_markers if marker in text_lower)
    
    if formal_count > informal_count:
        task_strengths.append("Tone is semi-formal, consistent")
    
    # Task errors
    for symbol in ['dev', '¶']:
        if symbol in error_counts_by_symbol:
            task_errors[symbol] = error_counts_by_symbol[symbol]
    
    # Generate task feedback
    if 'dev' in task_errors:
        task_feedback.append("Slightly underdeveloped conclusion - expand with more detailed stance")
    task_feedback.append("Strengthen conclusion by restating main points with a nuanced stance")
    
    # 5. Mechanics & Punctuation Assessment (X/5)
    mechanics_strengths = []
    mechanics_feedback = []
    mechanics_errors = {}
    
    if rubric.mechanics_punctuation >= 4.0:
        mechanics_strengths.append("Mostly accurate spelling and punctuation")
    
    # Mechanics errors
    for symbol in ['sp', 'punc', 'apos', 'delete', '^', 'tr', 'caps', '#', '¶', 'cs']:
        if symbol in error_counts_by_symbol:
            mechanics_errors[symbol] = error_counts_by_symbol[symbol]
    
    # Generate mechanics feedback
    if 'cs' in mechanics_errors:
        mechanics_feedback.append("Review comma use with dependent clauses - comma splice risk (cs)")
    mechanics_feedback.append("Proofread carefully for agreement/punctuation consistency")
    
    # Generate overall narrative following the exact format
    level = rubric.cefr_level
    sublevel = rubric.cefr_sublevel
    total_score = rubric.total_score
    
    # Create detailed narrative feedback
    narrative = f"This text demonstrates a solid {level} {sublevel} level. "
    
    if level == "B2":
        narrative += "The student can argue both sides of a complex issue, provide relevant examples, and maintain coherence across paragraphs. "
    elif level == "B1":
        narrative += "The student shows good intermediate skills with connected paragraphs and varied language. "
    elif level == "A2":
        narrative += "The student demonstrates basic connected text with simple structures. "
    else:
        narrative += "The student shows developing language skills. "
    
    if error_counts_by_symbol:
        narrative += f"Errors are noticeable but do not significantly reduce clarity. "
    
    # Next Steps section
    narrative += "\n\nNext Steps:\n"
    if 'agr' in error_counts_by_symbol:
        narrative += "• Improve subject–verb agreement and preposition accuracy.\n"
    if level in ["B1", "B2"]:
        narrative += "• Use advanced discourse markers (moreover, nonetheless) to refine cohesion.\n"
        narrative += "• Expand vocabulary with collocations and idiomatic expressions to move toward higher proficiency.\n"
    
    # Error Summary section
    if error_counts_by_symbol:
        narrative += "\nError Summary:\n"
        category_summary = {}
        for symbol in correction_symbols:
            category = symbol.category
            if category not in category_summary:
                category_summary[category] = {}
            cat_symbol = symbol.symbol
            category_summary[category][cat_symbol] = category_summary[category].get(cat_symbol, 0) + 1
        
        for category, symbols in category_summary.items():
            symbol_list = [f"{sym} ×{count}" for sym, count in symbols.items()]
            narrative += f"• {category}: {', '.join(symbol_list)}\n"
    
    return AssessmentFeedback(
        vocabulary=CategoryAssessment(
            strengths=vocab_strengths,
            range_or_structures=vocab_range,
            errors_by_symbol=vocab_errors,
            score=rubric.vocabulary_range_appropriacy,
            feedback=vocab_feedback
        ),
        grammar=CategoryAssessment(
            strengths=grammar_strengths,
            range_or_structures="Uses contrastive structures" if grammar_strengths else "Basic structures",
            errors_by_symbol=grammar_errors,
            score=rubric.grammar_range_accuracy,
            feedback=grammar_feedback
        ),
        coherence_cohesion=CategoryAssessment(
            strengths=coherence_strengths,
            range_or_structures="Advanced connectors" if advanced_count > 1 else "Basic connectors",
            errors_by_symbol=coherence_errors,
            score=rubric.coherence_cohesion,
            feedback=coherence_feedback
        ),
        task_fulfillment_register=CategoryAssessment(
            strengths=task_strengths,
            range_or_structures="Semi-formal, consistent" if formal_count > informal_count else "Mixed register",
            errors_by_symbol=task_errors,
            score=rubric.task_fulfillment_register,
            feedback=task_feedback
        ),
        mechanics_punctuation=CategoryAssessment(
            strengths=mechanics_strengths,
            errors_by_symbol=mechanics_errors,
            score=rubric.mechanics_punctuation,
            feedback=mechanics_feedback
        ),
        total_score=total_score,
        cefr_level=level,
        cefr_sublevel=sublevel,
        error_summary_by_symbol=error_counts_by_symbol,
        correction_symbols=correction_symbols,
        narrative_feedback=narrative
    )

def analyze_vocabulary_range_appropriacy(doc, text: str) -> tuple:
    """Analyze vocabulary range and appropriacy (0-5 points)"""
    if not doc:
        return 2.0, []
    
    # Calculate lexical diversity (Type-Token Ratio)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    unique_tokens = set(tokens)
    lexical_diversity = len(unique_tokens) / len(tokens) if tokens else 0
    
    # Count advanced vocabulary (longer words, complex forms)
    advanced_words = [token for token in doc if len(token.text) > 6 and token.pos_ in ["NOUN", "VERB", "ADJ"]]
    advanced_ratio = len(advanced_words) / len([t for t in doc if t.is_alpha]) if doc else 0
    
    # Calculate score based on diversity and complexity
    diversity_score = min(5.0, lexical_diversity * 7.5)  # Scale to 0-5
    complexity_score = min(5.0, advanced_ratio * 10)    # Scale to 0-5
    
    vocabulary_score = (diversity_score + complexity_score) / 2
    
    errors = []
    if lexical_diversity < 0.3:
        errors.append({"type": "vocabulary", "symbol": "rep", "message": "Limited vocabulary range - try using more varied words"})
    if advanced_ratio < 0.1:
        errors.append({"type": "vocabulary", "symbol": "ww", "message": "Use more sophisticated vocabulary for higher levels"})
    
    return vocabulary_score, errors

def analyze_grammar_range_accuracy(doc, text: str) -> tuple:
    """Analyze grammar range and accuracy (0-5 points)"""
    if not doc:
        return 2.0, []
    
    # Count sentence structures
    sentences = list(doc.sents)
    sentence_lengths = [len(sent) for sent in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    
    # Analyze tense variety
    verb_tenses = [token.tag_ for token in doc if token.pos_ == "VERB"]
    tense_variety = len(set(verb_tenses)) / len(verb_tenses) if verb_tenses else 0
    
    # Check for complex structures
    complex_structures = 0
    for token in doc:
        if token.dep_ in ["csubj", "ccomp", "advcl", "relcl"]:  # Complex clauses
            complex_structures += 1
    
    complex_ratio = complex_structures / len(sentences) if sentences else 0
    
    # Basic error detection
    errors = []
    
    # Subject-verb agreement check (basic)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            if token.tag_ in ["NNS", "NNPS"] and token.head.tag_ in ["VBZ"]:
                errors.append({
                    "type": "grammar", 
                    "symbol": "agr",
                    "position": token.idx,
                    "message": "Possible subject-verb agreement error"
                })
    
    # Calculate grammar score
    length_score = min(5.0, avg_sentence_length / 4)  # Reward longer sentences
    variety_score = min(5.0, tense_variety * 10)      # Reward tense variety
    complexity_score = min(5.0, complex_ratio * 15)   # Reward complex structures
    
    grammar_score = (length_score + variety_score + complexity_score) / 3
    
    # Reduce score for errors
    error_penalty = min(2.0, len(errors) * 0.3)
    grammar_score = max(0, grammar_score - error_penalty)
    
    return grammar_score, errors

def analyze_coherence_cohesion(doc, text: str) -> tuple:
    """Analyze coherence and cohesion (0-5 points)"""
    if not doc:
        return 2.0, []
    
    # Common discourse markers
    discourse_markers = {
        "addition": ["furthermore", "moreover", "additionally", "also", "besides"],
        "contrast": ["however", "nevertheless", "nonetheless", "although", "despite"],
        "sequence": ["firstly", "secondly", "finally", "then", "next"],
        "result": ["therefore", "consequently", "thus", "hence", "as a result"],
        "example": ["for example", "for instance", "such as", "namely"]
    }
    
    found_markers = []
    marker_types = set()
    
    text_lower = text.lower()
    for category, markers in discourse_markers.items():
        for marker in markers:
            if marker in text_lower:
                found_markers.append(marker)
                marker_types.add(category)
    
    # Analyze pronoun reference and repetition
    pronouns = [token.text for token in doc if token.pos_ == "PRON"]
    pronoun_ratio = len(pronouns) / len([t for t in doc if t.is_alpha]) if doc else 0
    
    # Calculate cohesion score
    marker_score = min(5.0, len(marker_types) * 1.0)      # Variety of connectors
    pronoun_score = min(5.0, pronoun_ratio * 20)          # Appropriate pronoun use
    
    coherence_score = (marker_score + pronoun_score) / 2
    
    errors = []
    if len(found_markers) == 0:
        errors.append({"type": "coherence", "symbol": "coh", "message": "Add connecting words to improve text flow"})
    
    return coherence_score, found_markers

def analyze_task_fulfillment_register(doc, text: str, task_type: Optional[str] = "general") -> tuple:
    """Analyze task fulfillment and register appropriacy (0-5 points)"""
    if not doc:
        return 2.0, {}
    
    # Analyze register features
    formal_markers = ["therefore", "furthermore", "consequently", "moreover", "nevertheless"]
    informal_markers = ["so", "but", "and then", "well", "you know"]
    
    formal_count = sum(1 for marker in formal_markers if marker in text.lower())
    informal_count = sum(1 for marker in informal_markers if marker in text.lower())
    
    # Word choice analysis
    word_lengths = [len(token.text) for token in doc if token.is_alpha]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    # Sentence variety
    sentences = list(doc.sents)
    question_count = sum(1 for sent in sentences if sent.text.strip().endswith('?'))
    exclamation_count = sum(1 for sent in sentences if sent.text.strip().endswith('!'))
    
    # Calculate register appropriacy
    formality_ratio = formal_count / (formal_count + informal_count + 1)
    
    # Task fulfillment score (generic for now - could be enhanced per task type)
    word_count = len([t for t in doc if t.is_alpha])
    length_appropriacy = 1.0 if 50 <= word_count <= 500 else 0.7
    
    register_score = (formality_ratio * 3 + length_appropriacy * 2)
    register_score = min(5.0, register_score)
    
    analysis = {
        "formality_ratio": formality_ratio,
        "avg_word_length": avg_word_length,
        "sentence_variety": len(set([len(sent) for sent in sentences])),
        "register_markers": {"formal": formal_count, "informal": informal_count}
    }
    
    return register_score, analysis

def analyze_mechanics_punctuation(text: str) -> tuple:
    """Analyze mechanics and punctuation (0-5 points)"""
    errors = []
    
    # Basic punctuation checks
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Check for missing capitalization at sentence start
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence and not sentence[0].isupper():
            errors.append({
                "type": "mechanics",
                "symbol": "caps",
                "message": "Capitalize first word of sentence"
            })
    
    # Check for comma splices (basic detection)
    comma_splice_pattern = r',\s*[a-z].*[.!?]'
    if re.search(comma_splice_pattern, text):
        errors.append({
            "type": "mechanics",
            "symbol": "cs",
            "message": "Possible comma splice - consider using semicolon or period"
        })
    
    # Check for double spaces
    if '  ' in text:
        errors.append({
            "type": "mechanics", 
            "symbol": "sp",
            "message": "Remove extra spaces"
        })
    
    # Calculate mechanics score
    base_score = 5.0
    error_penalty = min(3.0, len(errors) * 0.4)
    mechanics_score = max(0, base_score - error_penalty)
    
    return mechanics_score, errors

def analyze_vocabulary_range_appropriacy_enhanced(doc, text: str, features: Dict) -> float:
    """Enhanced vocabulary analysis using NLP features"""
    if not doc:
        return 2.0
    
    lexical_diversity = features.get('lexical_diversity', 0)
    advanced_ratio = features.get('advanced_vocabulary_ratio', 0)
    
    # Enhanced scoring with CEFR analyzer
    if cefr_analyzer:
        words = [token.text.lower() for token in doc if token.is_alpha]
        cefr_levels = []
        for word in words[:50]:  # Limit for performance
            try:
                level = cefr_analyzer.get_average_word_level_CEFR(word)
                if level:
                    # Convert CEFR level to numeric if it's not already
                    if isinstance(level, (int, float)):
                        cefr_levels.append(level)
                    else:
                        # Handle CEFRLevel object or string
                        level_str = str(level).upper() if hasattr(level, '__str__') else level
                        level_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
                        numeric_level = level_map.get(level_str, 3)
                        cefr_levels.append(numeric_level)
            except:
                continue
        
        # Calculate average CEFR level
        avg_cefr = sum(cefr_levels) / len(cefr_levels) if cefr_levels else 3
        cefr_score = min(5.0, avg_cefr * 1.0)  # Scale appropriately
    else:
        cefr_score = 3.0
    
    # Combine scores
    diversity_score = min(5.0, lexical_diversity * 8)
    complexity_score = min(5.0, advanced_ratio * 12)
    
    final_score = (diversity_score + complexity_score + cefr_score) / 3
    return max(0.5, min(5.0, final_score))

def analyze_grammar_range_accuracy_enhanced(doc, text: str, features: Dict) -> float:
    """Enhanced grammar analysis with error detection"""
    if not doc:
        return 2.0
    
    # Base grammar complexity analysis
    complexity_ratio = features.get('sentence_complexity_ratio', 0)
    error_rate = features.get('error_rate', 0)
    
    # Grammar complexity score based on sentence structures
    complexity_score = min(5.0, complexity_ratio * 8)
    
    # Accuracy score based on error rate
    accuracy_score = max(1.0, 5.0 - (error_rate * 20))
    
    # Sentence variety analysis
    sentences = list(doc.sents)
    sentence_lengths = [len(sent) for sent in sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    length_variety = len(set(sentence_lengths)) / len(sentence_lengths) if sentence_lengths else 0
    
    variety_score = min(5.0, (avg_length / 15) * 5 + length_variety * 3)
    
    final_score = (complexity_score + accuracy_score + variety_score) / 3
    return max(0.5, min(5.0, final_score))

def analyze_coherence_cohesion_enhanced(doc, text: str, features: Dict) -> float:
    """Enhanced coherence analysis with transition markers"""
    if not doc:
        return 2.0
    
    coherence_density = features.get('coherence_marker_density', 0)
    
    # Coherence markers analysis
    coherence_score = min(5.0, coherence_density * 15)
    
    # Paragraph structure analysis
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        paragraph_score = min(5.0, len(paragraphs) * 1.5)
    else:
        paragraph_score = 2.0
    
    # Logical flow analysis (simplified)
    sentences = list(doc.sents)
    transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless']
    transition_count = sum(1 for sent in sentences for word in transition_words if word in sent.text.lower())
    
    flow_score = min(5.0, (transition_count / len(sentences)) * 20) if sentences else 0
    
    final_score = (coherence_score + paragraph_score + flow_score) / 3
    return max(0.5, min(5.0, final_score))

def analyze_task_fulfillment_register_enhanced(doc, text: str, features: Dict) -> float:
    """Enhanced task fulfillment and register analysis"""
    if not doc:
        return 2.0
    
    register_consistency = features.get('register_consistency_score', 0)
    
    # Word count adequacy
    word_count = len(text.split())
    length_score = min(5.0, word_count / 50)  # 50 words = 1 point, 250+ words = 5 points
    
    # Register appropriateness
    register_score = min(5.0, register_consistency * 5)
    
    # Task development (idea elaboration)
    sentences = list(doc.sents)
    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
    development_score = min(5.0, avg_sentence_length / 15 * 5)
    
    final_score = (length_score + register_score + development_score) / 3
    return max(0.5, min(5.0, final_score))

def analyze_mechanics_punctuation_enhanced(doc, text: str, features: Dict) -> float:
    """Enhanced mechanics analysis with correction symbols"""
    if not doc:
        return 2.0
    
    correction_symbols = features.get('correction_symbols', [])
    
    # Error analysis by category
    mechanics_errors = [s for s in correction_symbols if s.category == 'Mechanics & Punctuation']
    grammar_errors = [s for s in correction_symbols if s.category == 'Grammar & Usage']
    
    # Calculate scores based on error density
    word_count = len(text.split())
    mechanics_error_rate = len(mechanics_errors) / word_count if word_count > 0 else 0
    
    # Base score minus penalties
    base_score = 5.0
    error_penalty = min(4.0, mechanics_error_rate * 100)  # Heavy penalty for mechanics errors
    
    final_score = max(0.5, base_score - error_penalty)
    return min(5.0, final_score)

@app.post("/score", response_model=ScoringResponse)
async def score_text(
    request: TextAnalysisRequest,
    authorization: Optional[str] = Header(None)
):
    """Score text using 5-category CEFR rubric"""
    
    # Language model check removed for free hosting compatibility
    
    # Basic validation
    if len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short for analysis")
    
    if len(request.text) > 10000:  # 10KB limit
        raise HTTPException(status_code=400, detail="Text too long for analysis")
    
    start_time = time.time()
    
    # Process text with spaCy
    doc = nlp(request.text[:5000])  # Limit for performance
    
    # Analyze each category
    vocab_score, vocab_errors = analyze_vocabulary_range_appropriacy(doc, request.text)
    grammar_score, grammar_errors = analyze_grammar_range_accuracy(doc, request.text)
    coherence_score, coherence_markers = analyze_coherence_cohesion(doc, request.text)
    task_score, register_analysis = analyze_task_fulfillment_register(doc, request.text, request.task_type or "general")
    mechanics_score, mechanics_errors = analyze_mechanics_punctuation(request.text)
    
    # Calculate totals
    total_score = vocab_score + grammar_score + coherence_score + task_score + mechanics_score
    cefr_level, cefr_sublevel = get_cefr_level(total_score)
    
    # Compile evidence
    tokens = [token for token in doc if token.is_alpha]
    sentences = list(doc.sents)
    
    all_errors = vocab_errors + grammar_errors + mechanics_errors
    
    evidence = AnalysisEvidence(
        word_count=len(tokens),
        sentence_count=len(sentences),
        avg_sentence_length=len(tokens) / len(sentences) if sentences else 0,
        lexical_diversity=len(set([t.lemma_.lower() for t in tokens])) / len(tokens) if tokens else 0,
        advanced_vocabulary_ratio=len([t for t in tokens if len(t.text) > 6]) / len(tokens) if tokens else 0,
        grammar_errors=grammar_errors,
        coherence_markers=coherence_markers,
        punctuation_errors=mechanics_errors,
        register_analysis=register_analysis
    )
    
    rubric = RubricScore(
        vocabulary_range_appropriacy=round(vocab_score, 1),
        grammar_range_accuracy=round(grammar_score, 1),
        coherence_cohesion=round(coherence_score, 1),
        task_fulfillment_register=round(task_score, 1),
        mechanics_punctuation=round(mechanics_score, 1),
        total_score=round(total_score, 1),
        cefr_level=cefr_level,
        cefr_sublevel=cefr_sublevel
    )
    
    processing_time = time.time() - start_time
    
    return ScoringResponse(
        rubric=rubric,
        evidence=evidence,
        processing_time=round(processing_time, 3)
    )

@app.post("/dictionary/analyze")
async def analyze_word(request: WordAnalysisRequest):
    """Comprehensive word analysis using NLTK WordNet and CEFR analyzer"""
    # Language model check removed for free hosting compatibility
    
    word = request.word.lower().strip()
    
    try:
        # Get CEFR level using cefrpy
        cefr_level = None
        if cefr_analyzer:
            try:
                cefr_level = cefr_analyzer.get_average_word_level_CEFR(word)
            except Exception as e:
                print(f"CEFR analysis error for '{word}': {e}")
        
        # Get WordNet data
        definitions = []
        synonyms = set()
        antonyms = set()
        pos_tags = set()
        examples = []
        
        synsets = wn.synsets(word)
        
        for synset in synsets[:3]:  # Limit to top 3 meanings
            # Definition
            definition = {
                "definition": synset.definition(),
                "part_of_speech": synset.pos(),
                "synset_name": synset.name()
            }
            definitions.append(definition)
            pos_tags.add(synset.pos())
            
            # Examples
            if synset.examples() and request.include_examples:
                examples.extend(synset.examples()[:2])
            
            # Synonyms and antonyms
            if request.include_synonyms:
                for lemma in synset.lemmas()[:5]:  # Limit to 5 per synset
                    synonym_name = lemma.name().replace('_', ' ')
                    if synonym_name != word:
                        synonyms.add(synonym_name)
                    
                    # Antonyms
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name().replace('_', ' '))
        
        # SpaCy analysis for additional data
        doc = nlp(word)
        token = doc[0] if doc else None
        frequency_score = None
        
        if token:
            # Use token frequency as a proxy for word difficulty
            frequency_score = float(token.prob) if hasattr(token, 'prob') and token.prob else None
        
        # Convert POS tags to readable format
        pos_readable = []
        pos_mapping = {
            'n': 'noun',
            'v': 'verb', 
            'a': 'adjective',
            's': 'adjective',  # satellite adjective
            'r': 'adverb'
        }
        
        for pos in pos_tags:
            readable = pos_mapping.get(pos, pos)
            if readable not in pos_readable:
                pos_readable.append(readable)
        
        return WordAnalysisResponse(
            word=word,
            cefr_level=cefr_level,
            definitions=definitions,
            synonyms=list(synonyms)[:10],  # Limit to 10 synonyms
            antonyms=list(antonyms)[:10],   # Limit to 10 antonyms
            part_of_speech=pos_readable,
            examples=examples[:5],  # Limit to 5 examples
            pronunciation=None,  # Could be added with additional libraries
            frequency_score=frequency_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/dictionary/enrich/{word}")
async def enrich_word(word: str):
    """Enhanced word information using dynamic dictionary libraries"""
    # Language model check removed for free hosting compatibility
    
    word = word.lower().strip()
    
    try:
        # Get CEFR level using cefrpy
        cefr_level = "Unknown"
        if cefr_analyzer:
            try:
                cefr_level = cefr_analyzer.get_average_word_level_CEFR(word) or "Unknown"
            except Exception as e:
                print(f"CEFR analysis error for '{word}': {e}")
        
        # SpaCy analysis
        doc = nlp(word)
        token = doc[0] if doc else None
        
        # WordNet quick lookup
        synsets = wn.synsets(word)
        primary_definition = synsets[0].definition() if synsets else "No definition found"
        
        return {
            "word": word,
            "lemma": token.lemma_ if token else word,
            "pos": token.pos_ if token else "UNKNOWN",
            "tag": token.tag_ if token else "UNKNOWN",
            "is_alpha": token.is_alpha if token else True,
            "is_stop": token.is_stop if token else False,
            "cefr_level": cefr_level,
            "definition": primary_definition,
            "synset_count": len(synsets),
            "has_multiple_meanings": len(synsets) > 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word enrichment error: {str(e)}")

@app.get("/dictionary/bulk-cefr")
async def bulk_cefr_analysis(words: str):
    """Analyze CEFR levels for multiple words (comma-separated)"""
    if not cefr_analyzer:
        raise HTTPException(status_code=500, detail="CEFR analyzer not available")
    
    word_list = [w.strip().lower() for w in words.split(',') if w.strip()]
    
    if len(word_list) > 50:  # Limit bulk requests
        raise HTTPException(status_code=400, detail="Too many words. Maximum 50 words per request.")
    
    results = {}
    
    for word in word_list:
        try:
            level = cefr_analyzer.get_average_word_level_CEFR(word)
            results[word] = level or "Unknown"
        except Exception as e:
            results[word] = "Error"
            print(f"CEFR analysis error for '{word}': {e}")
    
    return {
        "words_analyzed": len(word_list),
        "results": results,
        "analyzer_status": "active"
    }

@app.post("/score/enhanced")
async def score_text_enhanced(request: TextAnalysisRequest) -> EnhancedScoringResponse:
    """Enhanced scoring with comprehensive assessment flow"""
    start_time = time.time()
    
    text = request.text.strip()
    
    # Step 1: Text validity checks
    text_validity = check_text_validity(text, request.min_words or 50)
    
    if not text_validity['overall_valid']:
        # For invalid texts, return basic assessment with specific feedback
        return EnhancedScoringResponse(
            rubric=RubricScore(
                vocabulary_range_appropriacy=1.0,
                grammar_range_accuracy=1.0,
                coherence_cohesion=1.0,
                task_fulfillment_register=1.0,
                mechanics_punctuation=1.0,
                total_score=5.0,
                cefr_level="A1",
                cefr_sublevel="low"
            ),
            evidence=AnalysisEvidence(
                word_count=len(text.split()),
                sentence_count=len(re.split(r'[.!?]+', text)),
                avg_sentence_length=0,
                lexical_diversity=0,
                advanced_vocabulary_ratio=0,
                grammar_errors=[],
                coherence_markers=[],
                punctuation_errors=[],
                register_analysis={}
            ),
            feedback=AssessmentFeedback(
                vocabulary=CategoryAssessment(
                    strengths=[],
                    range_or_structures="basic",
                    errors_by_symbol={},
                    score=1.0,
                    feedback=["Expand vocabulary range"]
                ),
                grammar=CategoryAssessment(
                    strengths=[],
                    range_or_structures="incomplete",
                    errors_by_symbol={},
                    score=1.0,
                    feedback=["Complete sentences required"]
                ),
                coherence_cohesion=CategoryAssessment(
                    strengths=[],
                    range_or_structures="no connectors",
                    errors_by_symbol={},
                    score=1.0,
                    feedback=["Include connecting words and transitions"]
                ),
                task_fulfillment_register=CategoryAssessment(
                    strengths=[],
                    range_or_structures="inadequate",
                    errors_by_symbol={},
                    score=1.0,
                    feedback=["Use appropriate academic register"]
                ),
                mechanics_punctuation=CategoryAssessment(
                    strengths=[],
                    errors_by_symbol={},
                    score=1.0,
                    feedback=["Ensure proper capitalization and punctuation"]
                ),
                total_score=5.0,
                cefr_level="A1",
                cefr_sublevel="low",
                error_summary_by_symbol={},
                correction_symbols=[],
                narrative_feedback="Text does not meet minimum requirements for assessment. Please expand your response to at least 50 words with complete sentences."
            ),
            text_validity=text_validity,
            processing_time=time.time() - start_time
        )
    
    # Step 2: Enhanced NLP analysis (with fallback for free hosting)
    if nlp:
        doc = nlp(text)
        enhanced_features = analyze_enhanced_nlp_features(doc, text)
    else:
        # Fallback analysis without spaCy for free hosting
        doc = None
        enhanced_features = {
            'avg_sentence_length': len(text.split()) / max(1, len(re.split(r'[.!?]+', text))),
            'lexical_diversity': len(set(text.lower().split())) / max(1, len(text.split())),
            'advanced_vocabulary_ratio': len([w for w in text.split() if len(w) > 6]) / max(1, len(text.split())),
            'register_consistency_score': 0.7,
            'error_rate': 0.1,
            'correction_symbols': []
        }
    
    # Step 3: Correction symbol analysis
    correction_symbols = analyze_correction_symbols(text)
    
    # Step 4: Score calculation with enhanced features
    vocab_score = analyze_vocabulary_range_appropriacy_enhanced(doc, text, enhanced_features)
    grammar_score = analyze_grammar_range_accuracy_enhanced(doc, text, enhanced_features)
    coherence_score = analyze_coherence_cohesion_enhanced(doc, text, enhanced_features)
    task_score = analyze_task_fulfillment_register_enhanced(doc, text, enhanced_features)
    mechanics_score = analyze_mechanics_punctuation_enhanced(doc, text, enhanced_features)
    
    total_score = vocab_score + grammar_score + coherence_score + task_score + mechanics_score
    cefr_level, cefr_sublevel = get_cefr_level(total_score)
    
    # Step 5: Generate comprehensive feedback
    rubric = RubricScore(
        vocabulary_range_appropriacy=round(vocab_score, 1),
        grammar_range_accuracy=round(grammar_score, 1),
        coherence_cohesion=round(coherence_score, 1),
        task_fulfillment_register=round(task_score, 1),
        mechanics_punctuation=round(mechanics_score, 1),
        total_score=round(total_score, 1),
        cefr_level=cefr_level,
        cefr_sublevel=cefr_sublevel
    )
    
    feedback = generate_comprehensive_feedback(rubric, enhanced_features, correction_symbols, text, doc)
    
    # Evidence with enhanced features
    evidence = AnalysisEvidence(
        word_count=len(text.split()),
        sentence_count=len(list(doc.sents)) if doc else len(re.split(r'[.!?]+', text)),
        avg_sentence_length=enhanced_features.get('avg_sentence_length', 0),
        lexical_diversity=enhanced_features.get('lexical_diversity', 0),
        advanced_vocabulary_ratio=enhanced_features.get('advanced_vocabulary_ratio', 0),
        grammar_errors=[],
        coherence_markers=[],
        punctuation_errors=[],
        register_analysis={
            'consistency_score': enhanced_features.get('register_consistency_score', 0),
            'error_rate': enhanced_features.get('error_rate', 0)
        }
    )
    
    processing_time = time.time() - start_time
    
    return EnhancedScoringResponse(
        rubric=rubric,
        evidence=evidence,
        feedback=feedback,
        text_validity=text_validity,
        processing_time=round(processing_time, 3)
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "spacy_available": nlp is not None,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Render uses $PORT)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"=== Starting server on port {port} ===")
    print(f"Python version: {sys.version}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        log_level="info"
    )
