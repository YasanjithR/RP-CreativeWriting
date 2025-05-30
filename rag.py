# import os
# from dotenv import load_dotenv, find_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from itertools import permutations

# # Load environment variables
# dotenv_path = find_dotenv()
# load_dotenv(dotenv_path)

# # Initialize Pinecone with new API
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# # Initialize embedding and LLM
# embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# def create_pinecone_index(index_name="story", dimension=1536):
#     """
#     Create a Pinecone index for storing story embeddings
    
#     Args:
#         index_name (str): Name of the Pinecone index
#         dimension (int): Embedding dimension (1536 for text-embedding-ada-002)
#     """
#     # Check if index exists
#     existing_indexes = pc.list_indexes().names()
    
#     if index_name not in existing_indexes:
#         # Create serverless index
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric='cosine',
#             spec=ServerlessSpec(
#                 cloud='aws',
#                 region='us-west-2'
#             )
#         )
    
#     return pc.Index(index_name)

# def load_and_process_stories(pdf_path='./data/CreativeStoryData.pdf'):
#     """
#     Load PDF, split into chunks, and extract story metadata
    
#     Returns:
#         list: Processed story data
#     """
#     # Load PDF
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
    
#     # Split documents
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, 
#         chunk_overlap=200
#     )
#     texts = text_splitter.split_documents(docs)
    
#     # Process story metadata
#     processed_stories = []
#     for text_chunk in texts:
#         stories = text_chunk.page_content.split('---')
#         for story in stories:
#             story_data = parse_story_metadata(story)
#             if story_data:
#                 processed_stories.append(story_data)
    
#     return processed_stories

# def parse_story_metadata(story_text):
#     """
#     Parse individual story metadata from text
    
#     Args:
#         story_text (str): Raw story text
    
#     Returns:
#         dict: Parsed story metadata
#     """
#     lines = story_text.strip().split('\n')
#     story_data = {}
    
#     for line in lines:
#         if ':' in line:
#             key, value = line.split(':', 1)
#             key = key.strip().lower().replace(' ', '_')
#             story_data[key] = value.strip()
    
#     return story_data if story_data else None

# def index_stories_in_pinecone(index, stories, embedding_model):
#     """
#     Generate embeddings and index stories in Pinecone
    
#     Args:
#         index (pinecone.Index): Pinecone index
#         stories (list): List of story dictionaries
#         embedding_model: Embedding model
#     """
#     for story in stories:
#         # Create text to embed
#         text_to_embed = f"{story.get('title', '')} {story.get('description', '')}"
#         embedding = embedding_model.embed_query(text_to_embed)
        
#         # Prepare metadata
#         metadata = {k: v for k, v in story.items() if v}
        
#         # Upsert into Pinecone
#         index.upsert([
#             (story.get('story_id', str(hash(text_to_embed))), 
#              embedding, 
#              metadata)
#         ])

# def retrieve_similar_stories(index, embedding_model, query, top_k=3):
#     """
#     Retrieve similar stories from Pinecone
    
#     Args:
#         index (pinecone.Index): Pinecone index
#         embedding_model: Embedding model
#         query (str): User query
#         top_k (int): Number of similar stories to retrieve
    
#     Returns:
#         list: Retrieved story contexts
#     """
#     # Generate query embedding
#     query_embedding = embedding_model.embed_query(query)
    
#     # Query Pinecone
#     results = index.query(
#         vector=query_embedding, 
#         top_k=top_k, 
#         include_metadata=True
#     )
    
#     # Extract contexts
#     contexts = []
#     for match in results['matches']:
#         metadata = match['metadata']
#         context = f"Title: {metadata.get('title', 'N/A')}\n" \
#                   f"Description: {metadata.get('description', 'N/A')}"
#         contexts.append(context)
    
#     return contexts

# def generate_story_adventure(llm, contexts, user_query):
#     """
#     Generate a story adventure using LLM
    
#     Args:
#         llm: Language model
#         contexts (list): Retrieved story contexts
#         user_query (str): Original user query
    
#     Returns:
#         str: Generated story adventure
#     """
#     # Construct prompt
#     prompt_template = ChatPromptTemplate.from_template("""
#     You are a creative assistant helping to create stories for children.

#     Retrieved Story Contexts:
#     {context}

#     User Query: {input}

#     Generate an engaging and imaginative adventure that connects with the retrieved story contexts.
#     """)
    
#     # Create document chain
#     document_chain = create_stuff_documents_chain(llm, prompt_template)
    
#     # Generate response
#     response = document_chain.invoke({
#         "input": user_query,
#         "context": "\n\n".join(contexts)
#     })
    
#     return response





# ###################################################################
# def main():
#     # Create Pinecone index
#     index = create_pinecone_index()
    
#     # Load and process stories
#     stories = load_and_process_stories()
    
#     # Index stories in Pinecone
#     index_stories_in_pinecone(index, stories, embedding_model)
    
#     # Example query
#     user_query = "Create an adventure for a child exploring a new timeline"
    
#     # Retrieve similar stories
#     contexts = retrieve_similar_stories(index, embedding_model, user_query)
    
#     # Generate adventure
#     adventure = generate_story_adventure(llm, contexts, user_query)
    
#     print("Generated Adventure:", adventure)

# if __name__ == "__main__":
#     main()





import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from itertools import permutations
from langchain_core.documents import Document
# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
load_dotenv(override=True)

# Initialize Pinecone with current API
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
        # Create index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine'
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
        list: Retrieved story contexts as Document objects
    """
    # Generate query embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    
    # Extract contexts as Document objects
    contexts = []
    for match in results['matches']:
        metadata = match['metadata']
        context_text = f"Title: {metadata.get('title', 'N/A')}\n" \
                       f"Description: {metadata.get('description', 'N/A')}"
        contexts.append(Document(
            page_content=context_text,
            metadata=metadata
        ))
    
    return contexts

def generate_story_adventure(llm, contexts, user_query, short_story=False):
    """
    Generate a story adventure using LLM
    
    Args:
        llm: Language model
        contexts (list): Retrieved story contexts as Document objects
        user_query (str): Original user query
        short_story (bool): Whether to generate a short story with interchangeable sentences
    
    Returns:
        str: Generated story adventure
    """
    # Construct prompt
    if short_story:
        prompt_template = ChatPromptTemplate.from_template("""
        You are a creative assistant helping to create very simple, short stories for young children (ages 5-7).

        Retrieved Story Contexts:
        {context}

        User Query: {input}

        Generate a VERY SIMPLE short story with exactly 3-5 sentences.
        Important requirements:
        - Use extremely simple vocabulary suitable for young children
        - Keep sentences short (10-15 words maximum)
        - Use simple subject-verb-object structure
        - Avoid complex words, metaphors, or abstract concepts
        - Use only basic punctuation
        - The sentences should be completely interchangeable in order
        - Each sentence should be self-contained and make sense on its own
        - Focus on concrete, visual elements rather than abstract ideas
        - Use a reading level appropriate for ages 5-7
        
        Do NOT number the sentences or add a title.
        """)
    else:
        prompt_template = ChatPromptTemplate.from_template("""
        You are a creative assistant helping to create simple stories for children ages 7-10.

        Retrieved Story Contexts:
        {context}

        User Query: {input}

        Generate a simple, easy-to-understand story with the following characteristics:
        - Use simple vocabulary and short sentences (15-20 words maximum)
        - Avoid complex sentence structures or advanced vocabulary
        - Focus on concrete actions and descriptions rather than abstract ideas
        - Keep the narrative straightforward with a clear beginning, middle, and end
        - Use a reading level appropriate for ages 7-10
        - Limit to 10-15 sentences total
        """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Generate response
    response = document_chain.invoke({
        "input": user_query,
        "context": contexts
    })
    
    return response



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
    
    # Generate standard adventure
    adventure = generate_story_adventure(llm, contexts, user_query)
    print("\n=== Standard Adventure ===")
    print(adventure)
    
    # Generate short permutable story
    short_story_query = "Create a story about a magical forest with talking animals"
    short_story = generate_story_adventure(llm, contexts, short_story_query, short_story=True)
    print("\n=== Short Permutable Story ===")
    print(short_story)
    print("\nThis short story has interchangeable sentences suitable for permutation.")

if __name__ == "__main__":
    main()