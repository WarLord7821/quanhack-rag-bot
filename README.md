# AI-Powered Educational Enquiry Assistant 🤖🎓

A Retrieval-Augmented Generation (RAG) pipeline built to serve as an intelligent educational enquiry assistant for QuanHack Academy. This application ingests structured course data, stores it in a local vector database, and uses semantic search combined with a Large Language Model (LLM) to accurately answer student queries. It also features seamless WhatsApp integration via Twilio.

## ✨ Key Features

* **Custom RAG Architecture:** Utilizes raw `sentence-transformers` and `chromadb` for precise vector embeddings and semantic search, demonstrating a deep understanding of core RAG mechanics without relying on heavy frameworks like LangChain.
* **Ultra-Fast LLM Inference:** Powered by the Groq API utilizing the `llama3-70b-8192` model for near-instantaneous reasoning and response generation.
* **Smart Human Handoff:** Built-in intent recognition instantly routes users to human support when keywords like "talk to human" or "support" are detected, bypassing the LLM.
* **WhatsApp Integration:** A fully asynchronous FastAPI webhook handles real-time messaging via Twilio's Messaging API.
* **Grounded Responses:** Enforces a strict context window. The assistant is instructed to explicitly state if it does not know an answer outside of the embedded syllabus database, mitigating hallucinations.

## 🛠️ Tech Stack

* **Backend:** Python 3.x, FastAPI, Uvicorn
* **Vector Database:** ChromaDB (Persistent Local Client)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **LLM Engine:** Groq API (`llama3-70b-8192`)
* **Messaging & Webhook:** Twilio Messaging API

## 🏗️ Architecture

1. **Data Ingestion (`database_builder.py`):** Parses `courses.json`, combines course fields into comprehensive text strings, generates embeddings, and saves them locally in ChromaDB.
2. **AI Engine (`ai_engine.py`):** Handles query embedding, similarity search (top 2 results), human handoff logic, prompt construction, and LLM communication via Groq.
3. **Webhook (`main.py`):** FastAPI application that receives incoming Twilio POST requests, extracts the message body, queries the AI Engine, and returns valid TwiML XML.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/WarLord7821/quanhack-rag-bot.git
cd quanhack-rag-bot
```
### 2. Set Up the Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Environment Variables
Create a .env file in the root directory and add your Groq API key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Build the Vector Database
Ingest the courses.json file and create the local ChromaDB database (./chroma_data):
```bash
python database_builder.py
```
Note: This will download the all-MiniLM-L6-v2 embedding model on the first run.

### 6. Start the FastAPI Server
```bash
uvicorn main:app --reload
```
The server will start running locally at http://127.0.0.1:8000.

### 7. Connect Twilio via Ngrok
Expose your local port to the internet using Ngrok:
```bash
ngrok http 8000
```

Copy the generated HTTPS URL and append /webhook (e.g., https://your-url.ngrok.app/webhook). Paste this into your Twilio Sandbox settings for incoming messages.

Copyright (c) 2026 Rishabh Zambre. All rights reserved. This repository and its contents are provided solely for evaluation purposes as part of a technical interview process. Commercial use, reproduction, or distribution of this code without explicit permission is strictly prohibited.
