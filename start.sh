#!/bin/bash

# Download required NLTK data
python -c "
import nltk
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download error: {e}')
"

# Download spaCy model if not exists
python -m spacy download en_core_web_sm --quiet || echo "spaCy model download failed, using basic analysis"

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}