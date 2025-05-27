# Creative Writing Research Project

This project implements a creative writing assistant using RAG (Retrieval Augmented Generation) and adaptive story generation.

## Features

- Story generation based on creative prompts
- Retrieval-based context enrichment (RAG)
- Permutation validation for story structure
- Input validation with adaptive difficulty
- Adaptive story continuation based on user engagement

## Setup

### Regular Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows:
     ```
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source .venv/bin/activate
     ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Download required NLTK and spaCy resources:
   ```
   python setup_resources.py
   ```
6. Create a `.env` file with necessary API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

### Docker Setup

1. Make sure you have Docker and Docker Compose installed
2. Create a `.env` file with your API keys as described above
3. Build and start the container:
   ```
   docker-compose up -d
   ```
4. The API will be available at http://localhost:8000

## Running the Application

### Web API

To start the FastAPI server locally:

```
python api.py
```

The API will be available at http://localhost:8000, with documentation at http://localhost:8000/docs.

## API Endpoints

### RAG Story Generation

POST `/generate-story` - Generate a story using RAG

```json
{
  "characters": [{"name": "Lily", "description": "a brave astronaut"}],
  "setting": "space station",
  "theme": "friendship",
  "plot_elements": ["mysterious discovery", "teamwork"]
}
```

### Story Permutation Validation

POST `/permutations` - Find sensible permutations of a story

```json
{
  "story": "Once upon a time, there was a brave knight. He lived in a tall castle overlooking a vast kingdom. One day, he decided to go on a quest to save a captured princess.",
  "max_permutations": 5,
  "time_limit": 20
}
```

### Permutation Evaluation

POST `/evaluate-permutation` - Evaluate a specific permutation

```json
{
  "baseline_permutation": [
    "The sun rose over the mountains.",
    "The village began to wake up.",
    "Birds chirped in the distance."
  ],
  "valid_permutations": [
    ["The sun rose over the mountains.", "The village began to wake up.", "Birds chirped in the distance."],
    ["Birds chirped in the distance.", "The sun rose over the mountains.", "The village began to wake up."]
  ],
  "child_permutation": [
    "The sun rose over the mountains.",
    "Birds chirped in the distance.",
    "The village began to wake up."
  ]
}
```

### Adaptive Story Continuation

POST `/continue-story` - Generate adaptive story continuation

```json
{
  "previous_story": "In a world where time was a fabric that could be woven, young Aria discovered she had the rare ability to see temporal threads.",
  "creativity_score": 0.7,
  "validity_score": 0.6,
  "continuation_prompt": "Explore Aria's first attempt to manipulate time threads"
}
```

## Project Structure

- `rag.py`: Implements the retrieval-augmented generation system
- `permutationValidator.py`: Validates story permutations for coherence
- `inputValidator.py`: Evaluates user input and adaptively sets difficulty
- `adaptiveStoryGenerator.py`: Generates adaptive story continuations
- `api.py`: FastAPI implementation for web access
- `data/`: Contains story data for training and retrieval