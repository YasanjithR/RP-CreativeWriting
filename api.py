import os
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time

from rag import (
    create_pinecone_index,
    load_and_process_stories,
    index_stories_in_pinecone,
    retrieve_similar_stories,
    generate_story_adventure,
    embedding_model,
    llm
)

from permutationValidator import StoryPermutationValidator
from adaptiveStoryGenerator import NarrativeContinuationGenerator
from inputValidator import TextPermutationEvaluator
from inputValidator import TextPermutationEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="Creative Story Generator API",
    description="Generate creative stories for children using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the Pinecone index
pinecone_index = None
indexed = False

# Initialize the validators and generators once
story_validator = StoryPermutationValidator()
narrative_generator = NarrativeContinuationGenerator()

# Pydantic models for request and response
class StoryParameters(BaseModel):
    characters: List[Dict[str, str]] = Field(
        default=[],
        description="List of character names and descriptions"
    )
    setting: Optional[str] = Field(
        default=None, 
        description="Setting of the story (e.g., forest, space, underwater)"
    )
    theme: Optional[str] = Field(
        default=None,
        description="Theme of the story (e.g., friendship, adventure)"
    )
    plot_elements: Optional[List[str]] = Field(
        default=None,
        description="Key elements to include in the plot"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Any additional context or requirements for the story"
    )

class StoryResponse(BaseModel):
    story: str = Field(..., description="Generated story")
    used_contexts: List[Dict[str, str]] = Field(..., description="Story contexts used for generation")
    query: str = Field(..., description="Generated query used for retrieval")

class PermutationRequest(BaseModel):
    story: str = Field(..., description="Text of the story to analyze")
    max_permutations: int = Field(default=10, description="Maximum number of permutations to return")
    time_limit: int = Field(default=30, description="Maximum processing time in seconds")

class PermutationResponse(BaseModel):
    permutations: List[str] = Field(..., description="List of sensical story permutations")
    processing_time: float = Field(..., description="Processing time in seconds")
    permutations_evaluated: int = Field(..., description="Number of permutations evaluated")
    
class EvaluatePermutationRequest(BaseModel):
    baseline_permutation: List[str] = Field(..., description="Original order of sentences (baseline)")
    valid_permutations: List[List[str]] = Field(..., description="List of acceptable alternative orders")
    child_permutation: List[str] = Field(..., description="The permutation to evaluate")
    validity_weight: float = Field(default=0.7, description="Weight for validity score (0-1)")
    creativity_weight: float = Field(default=0.3, description="Weight for creativity score (0-1)")

class EvaluatePermutationResponse(BaseModel):
    validity_score: float = Field(..., description="How closely the permutation matches valid permutations (0-1)")
    creativity_score: float = Field(..., description="How different the permutation is from the baseline (0-1)")
    final_score: float = Field(..., description="Weighted combination of validity and creativity scores (0-1)")

# Dependency to ensure Pinecone index is initialized
async def get_pinecone_index():
    global pinecone_index, indexed
    if pinecone_index is None:
        pinecone_index = create_pinecone_index()
    
    return pinecone_index

# Background task to index stories
def index_stories_background():
    global pinecone_index, indexed
    if not indexed and pinecone_index is not None:
        stories = load_and_process_stories()
        index_stories_in_pinecone(pinecone_index, stories, embedding_model)
        indexed = True

# Startup event to initialize the Pinecone index
@app.on_event("startup")
async def startup_event():
    global pinecone_index
    pinecone_index = create_pinecone_index()

# Function to construct a query from story parameters
def construct_query(params: StoryParameters) -> str:
    query_parts = []
    
    # Add character information
    if params.characters:
        character_desc = ", ".join([f"{char.get('name', 'someone')} who is {char.get('description', '')}" 
                                  for char in params.characters])
        query_parts.append(f"characters: {character_desc}")
    
    # Add setting
    if params.setting:
        query_parts.append(f"set in {params.setting}")
    
    # Add theme
    if params.theme:
        query_parts.append(f"about {params.theme}")
    
    # Add plot elements
    if params.plot_elements:
        plot_desc = ", ".join(params.plot_elements)
        query_parts.append(f"including {plot_desc}")
    
    # Add additional context
    if params.additional_context:
        query_parts.append(params.additional_context)
    
    # Combine parts into a cohesive query
    if query_parts:
        query = "Create a story " + " ".join(query_parts)
    else:
        query = "Create an engaging adventure story for children"
    
    return query

# API endpoint to generate a story
@app.post("/generate-story", response_model=StoryResponse)
async def generate_story(
    params: StoryParameters,
    background_tasks: BackgroundTasks,
    pinecone_index=Depends(get_pinecone_index)
):
    # Start background indexing if not already done
    if not indexed:
        background_tasks.add_task(index_stories_background)
    
    # Construct query from parameters
    query = construct_query(params)
    
    try:
        # Retrieve relevant story contexts
        contexts = retrieve_similar_stories(pinecone_index, embedding_model, query)
        
        # Generate the story
        story = generate_story_adventure(llm, contexts, query)
        
        # Format used contexts for response
        used_contexts = []
        for context in contexts:
            title = context.metadata.get('title', 'Unknown')
            description = context.metadata.get('description', 'No description')
            used_contexts.append({"title": title, "description": description})
        
        return {
            "story": story,
            "used_contexts": used_contexts,
            "query": query
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

# API endpoint for story permutation validation
@app.post("/permutations", response_model=PermutationResponse)
async def generate_permutations(request: PermutationRequest):
    """
    Generate sensible permutations of a story
    
    Parameters:
    - story: Text of the story to analyze
    - max_permutations: Maximum number of permutations to return (default: 10)
    - time_limit: Maximum processing time in seconds (default: 30)
    
    Returns:
    - permutations: List of sensical story permutations
    - processing_time: Processing time in seconds
    - permutations_evaluated: Number of permutations evaluated
    """
    try:
        # Track start time for performance metrics
        start_time = time.time()
        
        # Generate permutations
        sensical_permutations = story_validator.generate_most_sensical_permutations(
            request.story, 
            max_permutations=request.max_permutations,
            time_limit=request.time_limit
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "permutations": sensical_permutations,
            "processing_time": round(processing_time, 2),
            "permutations_evaluated": len(story_validator._coherence_scores)
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating permutations: {str(e)}")

# API endpoint for evaluating permutations
@app.post("/evaluate-permutation", response_model=EvaluatePermutationResponse)
async def evaluate_permutation(request: EvaluatePermutationRequest):
    """
    Evaluate a specific permutation of sentences compared to baseline and valid permutations
    
    Parameters:
    - baseline_permutation: Original order of sentences
    - valid_permutations: List of acceptable alternative orders
    - child_permutation: The permutation to evaluate
    - validity_weight: Weight for validity score (default: 0.7)
    - creativity_weight: Weight for creativity score (default: 0.3)
    
    Returns:
    - validity_score: How closely the permutation matches valid permutations (0-1)
    - creativity_score: How different the permutation is from the baseline (0-1)
    - final_score: Weighted combination of validity and creativity scores (0-1)
    """
    try:
        # Create evaluator instance
        evaluator = TextPermutationEvaluator(
            request.baseline_permutation,
            request.valid_permutations
        )
        
        # Evaluate the permutation
        scores = evaluator.evaluate_permutation(
            request.child_permutation,
            validity_weight=request.validity_weight,
            creativity_weight=request.creativity_weight
        )
        
        return scores
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating permutation: {str(e)}")

# API endpoint for adaptive story continuation
class StoryContinuationRequest(BaseModel):
    previous_story: str = Field(..., description="The previous story text to continue from")
    creativity_score: float = Field(default=0.5, description="Previous creativity score (0-1)")
    validity_score: float = Field(default=0.5, description="Previous validity score (0-1)")
    continuation_prompt: str = Field(..., description="Specific prompt for continuing the story")

class StoryContinuationResponse(BaseModel):
    story_continuation: str = Field(..., description="Generated continuation of the story")
    complexity_level: int = Field(..., description="Calculated complexity level (1-5)")
    narrative_context: Dict = Field(..., description="Extracted narrative elements")
    complexity_history: Dict = Field(..., description="Complexity scoring history")

@app.post("/continue-story", response_model=StoryContinuationResponse)
async def continue_story(request: StoryContinuationRequest):
    """
    Generate a continuation of a story using adaptive complexity
    
    Parameters:
    - previous_story: The story text to continue from
    - creativity_score: Previous creativity score (0-1)
    - validity_score: Previous validity score (0-1)
    - continuation_prompt: Specific prompt for continuing the story
    
    Returns:
    - story_continuation: Generated continuation of the story
    - complexity_level: Calculated complexity level (1-5)
    - narrative_context: Extracted narrative elements
    - complexity_history: Complexity scoring history
    """
    try:
        # Create scores dictionary from individual parameters
        previous_scores = {
            "creativity_score": request.creativity_score,
            "validity_score": request.validity_score
        }
        
        # Generate story continuation with all parameters
        result = narrative_generator.generate_story_continuation(
            previous_story=request.previous_story,
            previous_scores=previous_scores,
            continuation_prompt=request.continuation_prompt
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story continuation: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "indexed": indexed}

# Run the application
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 