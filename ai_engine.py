import os
from typing import List
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================================
# Setup & Initialization
# ==========================================

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client
# The client automatically looks for the GROQ_API_KEY environment variable.
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")

# Initialize ChromaDB Persistent Client and retrieve the collection
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="courses_collection")

# Initialize the Sentence Transformer embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# Core RAG Functions
# ==========================================

def search_database(query: str) -> str:
    """
    Embeds the user's query and searches the ChromaDB vector database 
    for the top 2 most relevant documents.
    """
    # 1. Embed the query into a vector
    query_embedding: List[float] = embedding_model.encode(query).tolist()
    
    # 2. Query the Chroma collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    # 3. Extract and combine the retrieved documents into a single string
    # Chroma returns documents as a nested list: [['doc1', 'doc2']]
    retrieved_documents: List[str] = results.get('documents', [[]])[0]
    
    # Combine the documents into a single context string
    combined_context: str = "\n\n---\n\n".join(retrieved_documents)
    
    return combined_context

def generate_response(user_message: str) -> str:
    """
    Processes the user's message, checks for human handoff, retrieves context,
    and generates an AI response via Groq.
    """
    # 1. Human Handoff Check
    handoff_keywords: List[str] = ['talk to human', 'real person', 'support']
    user_message_lower: str = user_message.lower()
    
    if any(keyword in user_message_lower for keyword in handoff_keywords):
        return "I want to make sure you get the best help. Let me connect you with one of our human academic counselors. They will reach out shortly."
    
    # 2. Retrieve Context from Vector Database
    context: str = search_database(user_message)
    
    # 3. Construct System Prompt
    system_prompt: str = (
        "You are the QuanHack Academy Assistant. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, politely say you do not know. "
        "If the user seems interested in a course, proactively ask for their email address to send them a syllabus.\n\n"
        f"Context provided from database:\n{context}"
    )
    
    # 4. Call the Groq API (llama3-70b-8192)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2, # Lower temperature for grounded RAG responses
        )
        
        # 5. Return the generated string
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"An error occurred while communicating with the LLM: {str(e)}"


