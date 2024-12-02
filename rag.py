import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from itertools import permutations

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Initialize Pinecone with new API
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize embedding and LLM
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def create_pinecone_index(index_name="story", dimension=1536):
    """
    Create a Pinecone index for storing story embeddings
    
    Args:
        index_name (str): Name of the Pinecone index
        dimension (int): Embedding dimension (1536 for text-embedding-ada-002)
    """
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        # Create serverless index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
    
    return pc.Index(index_name)

def load_and_process_stories(pdf_path='./data/CreativeStoryData.pdf'):
    """
    Load PDF, split into chunks, and extract story metadata
    
    Returns:
        list: Processed story data
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)
    
    # Process story metadata
    processed_stories = []
    for text_chunk in texts:
        stories = text_chunk.page_content.split('---')
        for story in stories:
            story_data = parse_story_metadata(story)
            if story_data:
                processed_stories.append(story_data)
    
    return processed_stories

def parse_story_metadata(story_text):
    """
    Parse individual story metadata from text
    
    Args:
        story_text (str): Raw story text
    
    Returns:
        dict: Parsed story metadata
    """
    lines = story_text.strip().split('\n')
    story_data = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            story_data[key] = value.strip()
    
    return story_data if story_data else None

def index_stories_in_pinecone(index, stories, embedding_model):
    """
    Generate embeddings and index stories in Pinecone
    
    Args:
        index (pinecone.Index): Pinecone index
        stories (list): List of story dictionaries
        embedding_model: Embedding model
    """
    for story in stories:
        # Create text to embed
        text_to_embed = f"{story.get('title', '')} {story.get('description', '')}"
        embedding = embedding_model.embed_query(text_to_embed)
        
        # Prepare metadata
        metadata = {k: v for k, v in story.items() if v}
        
        # Upsert into Pinecone
        index.upsert([
            (story.get('story_id', str(hash(text_to_embed))), 
             embedding, 
             metadata)
        ])

def retrieve_similar_stories(index, embedding_model, query, top_k=3):
    """
    Retrieve similar stories from Pinecone
    
    Args:
        index (pinecone.Index): Pinecone index
        embedding_model: Embedding model
        query (str): User query
        top_k (int): Number of similar stories to retrieve
    
    Returns:
        list: Retrieved story contexts
    """
    # Generate query embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    
    # Extract contexts
    contexts = []
    for match in results['matches']:
        metadata = match['metadata']
        context = f"Title: {metadata.get('title', 'N/A')}\n" \
                  f"Description: {metadata.get('description', 'N/A')}"
        contexts.append(context)
    
    return contexts

def generate_story_adventure(llm, contexts, user_query):
    """
    Generate a story adventure using LLM
    
    Args:
        llm: Language model
        contexts (list): Retrieved story contexts
        user_query (str): Original user query
    
    Returns:
        str: Generated story adventure
    """
    # Construct prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are a creative assistant helping to create stories for children.

    Retrieved Story Contexts:
    {context}

    User Query: {input}

    Generate an engaging and imaginative adventure that connects with the retrieved story contexts.
    """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Generate response
    response = document_chain.invoke({
        "input": user_query,
        "context": "\n\n".join(contexts)
    })
    
    return response



################################################################

import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import networkx as nx
from transformers import pipeline

class StoryPermutationValidator:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Load spaCy language model for linguistic analysis
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize coherence checker
        self.coherence_checker = pipeline('text-classification', 
                                          model='facebook/bart-large-mnli')
        
        # Stopwords for filtering
        self.stop_words = set(stopwords.words('english'))

    def tokenize_story(self, story):
        """
        Tokenize the story into sentences
        
        Args:
            story (str): Full story text
        
        Returns:
            list: List of sentences
        """
        return sent_tokenize(story)

    def generate_permutations(self, sentences):
        """
        Generate all possible permutations of sentences
        
        Args:
            sentences (list): List of story sentences
        
        Returns:
            list: List of all possible sentence arrangements
        """
        return list(itertools.permutations(sentences))

    def check_grammatical_coherence(self, permutation):
        """
        Check basic grammatical coherence of a sentence permutation
        
        Args:
            permutation (tuple): Sequence of sentences
        
        Returns:
            float: Coherence score
        """
        # Join sentences to create a potential story
        story_text = ' '.join(permutation)
        
        # Use spaCy for linguistic analysis
        doc = self.nlp(story_text)
        
        # Check dependency parsing and sentence structure
        dependency_score = self._analyze_dependency_structure(doc)
        
        return dependency_score

    def _analyze_dependency_structure(self, doc):
        """
        Analyze the dependency parsing of the document
        
        Args:
            doc (spacy.tokens.Doc): Parsed document
        
        Returns:
            float: Dependency coherence score
        """
        # Create a dependency graph
        G = nx.DiGraph()
        
        # Add nodes and edges based on dependency parsing
        for token in doc:
            G.add_node(token.text, pos=token.pos_, dep=token.dep_)
            if token.head.text != token.text:
                G.add_edge(token.head.text, token.text)
        
        # Calculate graph metrics
        try:
            clustering = nx.average_clustering(G)
            connectivity = nx.average_degree_connectivity(G)
            avg_connectivity = sum(connectivity.values()) / len(connectivity)
            
            # Combine metrics
            coherence_score = (clustering + avg_connectivity) / 2
            return min(max(coherence_score, 0), 1)
        except:
            return 0.5

    def semantic_coherence_check(self, permutation):
        """
        Check semantic coherence using transformer-based NLI
        
        Args:
            permutation (tuple): Sequence of sentences
        
        Returns:
            float: Semantic coherence score
        """
        # Compare adjacent sentences for logical progression
        coherence_scores = []
        for i in range(len(permutation) - 1):
            try:
                result = self.coherence_checker(
                    permutation[i] + ' ' + permutation[i+1], 
                    hypothesis='These sentences are logically connected'
                )[0]
                
                # Convert label to numerical score
                if result['label'] == 'ENTAILMENT':
                    score = result['score']
                elif result['label'] == 'CONTRADICTION':
                    score = 1 - result['score']
                else:
                    score = 0.5
                
                coherence_scores.append(score)
            except:
                coherence_scores.append(0.5)
        
        # Return average coherence
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

    def validate_permutation(self, permutation, threshold=0.6):
        """
        Validate if a permutation makes sense
        
        Args:
            permutation (tuple): Sequence of sentences
            threshold (float): Minimum coherence threshold
        
        Returns:
            bool: Whether the permutation is sensical
        """
        # Check different coherence aspects
        grammatical_score = self.check_grammatical_coherence(permutation)
        semantic_score = self.semantic_coherence_check(permutation)
        
        # Combine scores
        total_score = (grammatical_score + semantic_score) / 2
        
        return total_score >= threshold

    def generate_sensical_permutations(self, story, max_permutations=10):
        """
        Generate sensical permutations of a story
        
        Args:
            story (str): Original story text
            max_permutations (int): Maximum number of sensical permutations to return
        
        Returns:
            list: List of sensical story permutations
        """
        # Tokenize the story
        sentences = self.tokenize_story(story)
        
        # Generate all permutations
        all_permutations = self.generate_permutations(sentences)
        
        # Filter sensical permutations
        sensical_permutations = []
        for perm in all_permutations:
            if self.validate_permutation(perm):
                sensical_permutations.append(' '.join(perm))
                
                # Stop if we've found enough permutations
                if len(sensical_permutations) >= max_permutations:
                    break
        
        return sensical_permutations

def main():
    # Example usage
    story = """
    Once upon a time, there was a brave knight. 
    He lived in a tall castle overlooking a vast kingdom. 
    One day, he decided to go on a quest to save a captured princess. 
    After many challenges, he finally rescued the princess from the dragon's lair. 
    The kingdom celebrated his heroic deed with a grand feast.
    """
    
    # Initialize the permutation validator
    validator = StoryPermutationValidator()
    
    # Generate sensical permutations
    sensical_permutations = validator.generate_sensical_permutations(story)
    
    # Print the permutations
    print("Sensical Story Permutations:")
    for i, perm in enumerate(sensical_permutations, 1):
        print(f"\nPermutation {i}:")
        print(perm)

if __name__ == "__main__":
    main()



###################################################################
def main():
    # Create Pinecone index
    index = create_pinecone_index()
    
    # Load and process stories
    stories = load_and_process_stories()
    
    # Index stories in Pinecone
    index_stories_in_pinecone(index, stories, embedding_model)
    
    # Example query
    user_query = "Create an adventure for a child exploring a new timeline"
    
    # Retrieve similar stories
    contexts = retrieve_similar_stories(index, embedding_model, user_query)
    
    # Generate adventure
    adventure = generate_story_adventure(llm, contexts, user_query)
    
    print("Generated Adventure:", adventure)

if __name__ == "__main__":
    main()