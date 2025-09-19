#!/bin/bash

# Check Python version
python --version

# Download required NLTK data
python -c "
import nltk
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download error: {e}')
"

# Start the FastAPI application
exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1