import json
import chromadb
from sentence_transformers import SentenceTransformer

def build_database():
    print("Starting the database building process...")

    # 1. Load the courses.json file
    try:
        with open("courses.json", "r", encoding="utf-8") as f:
            courses_data = json.load(f)
        print(f"Successfully loaded {len(courses_data)} courses from courses.json.")
    except FileNotFoundError:
        print("Error: courses.json not found. Please ensure it is in the same directory.")
        return

    # 2. Initialize the Sentence Transformer model
    # The all-MiniLM-L6-v2 model is highly efficient for generating semantic vectors.
    print("Loading the embedding model 'all-MiniLM-L6-v2' (this may take a moment to download on the first run)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Initialize ChromaDB Persistent Client
    # This will create a directory named 'chroma_data' locally to store the vectors on disk.
    print("Initializing ChromaDB persistent client at './chroma_data'...")
    client = chromadb.PersistentClient(path="./chroma_data")

    # 4. Create (or get) a collection
    collection_name = "courses_collection"
    print(f"Creating or retrieving the collection: '{collection_name}'...")
    # Using get_or_create prevents errors if you run this script multiple times
    collection = client.get_or_create_collection(name=collection_name)

    # Prepare lists to structure the data for ChromaDB ingestion
    documents = []
    metadatas = []
    ids = []
    embeddings = []

    print("Processing courses and generating vector embeddings...")

    # 5. Iterate through the JSON data
    for index, course in enumerate(courses_data):
        
        # Format the course details into a single comprehensive string.
        # This unified context is what the embedding model will convert into a mathematical vector.
        course_text = (
            f"Course Name: {course.get('course_name', 'N/A')}. "
            f"Duration: {course.get('duration', 'N/A')}. "
            f"Price: {course.get('price', 'N/A')}. "
            f"Prerequisites: {', '.join(course.get('prerequisites', []))}. "
            f"Target Audience: {course.get('target_audience', 'N/A')}. "
            f"Syllabus Summary: {course.get('syllabus_summary', 'N/A')}"
        )

        # Generate the vector embedding for this unified text
        # .tolist() converts the resulting numpy array into a standard Python list for ChromaDB
        print(f"  -> Generating embedding for: {course.get('course_name')}")
        vector = model.encode(course_text).tolist()

        # Append to our respective batch lists
        documents.append(course_text)
        metadatas.append({"course_name": course.get('course_name', 'Unknown')})
        ids.append(f"course_{index + 1}")
        embeddings.append(vector)

    # 6. Add the generated data into the ChromaDB collection
    print("Inserting documents, metadata, and embeddings into the ChromaDB collection...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print("Database successfully built and saved to the './chroma_data' directory!")

if __name__ == "__main__":
    build_database()