#!/bin/bash

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

# Start the FastAPI application with gunicorn
exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}