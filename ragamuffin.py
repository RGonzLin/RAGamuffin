# RAGamuffin by Rodrigo González Linares

import ollama
import os
import PyPDF2
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi



def extract_text_from_txt(txt_path):
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except:
        # Ignore the document if an error occurs
        return None

def extract_text_from_directory(directory):

    documents = {}
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            text = None
            if file.endswith(('.txt', '.md', '.py', '.sh', '.js')): # Add more extensions if needed!
                text = extract_text_from_txt(path)
            # Add text to dictionary only if text extraction was successful
            if text:
                documents[path] = text

    return documents

def get_youtube_id(url):
    
    # Finding the start index of the video ID
    start_index = url.find('v=') + 2

    return url[start_index:]

def extract_text_from_webpage(url):

    # Check if the URL is from YouTube
    if "youtube.com" in url:
        # Extracting the video id
        video_id = get_youtube_id(url)
        # Fetching the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combining all text parts from the transcript
        text = ' '.join([entry['text'] for entry in transcript])

    else:
        # Send request
        request = requests.get(url)
        # Parse the HTML content
        soup = BeautifulSoup(request.text, 'html.parser')
        # Extract all text from the webpage
        text = soup.get_text()

    return text



def RAGamuffin():

    default = input("Use default LLM (llama3), embedding model (all-minilm), and path (docs/)? (y/n): ")

    if default == "y":
        llm = "llama3"
        embedding_model = "all-minilm"
        docu_path = "docs/"
    
    else:
        # Get the user to setup the models and the documents path
        llm = input("LLM: ")
        embedding_model = input("Embedding model: ")
        docu_path = input("Path to documents : ")

    print("Extracting texts")

    # Extract text from the documents
    texts = extract_text_from_directory(docu_path)

    print("Embedding documents...")

    # Embed the documents
    embeddings = {}
    for document, text in texts.items(): 
        embeddings[document] = ollama.embeddings(model=embedding_model, 
                                                    prompt=text)['embedding']
        
    # Delete the texts to save memory
    del texts

    print(f"{len(embeddings)} documents embedded!")

    print("Indexing documents...")

    # Convert the dictionary values to a 2D numpy array
    doc_names = list(embeddings.keys())
    doc_embeddings = np.array(list(embeddings.values()), dtype=np.float32)

    # Normalize the embeddings to use cosine similarity
    faiss.normalize_L2(doc_embeddings)

    # Create a index
    index = faiss.IndexFlatIP(doc_embeddings.shape[1]) 

    # Add vectors to the index
    index.add(doc_embeddings)

    print("Done! Let's RAG!")

    # Initialize parameters
    rag = True # RAG is on by default
    k_docs = 1 # Number of documents to retrieve
    min_score = 0.0 # Minimum similarity score to retrieve a document, orthogonal by default
    rag_docs = True # Show the documents retrieved by RAG if True
    rag_flag = True # Flag to momentarily turn off RAG
    history = [{
                "role": "system",
                "content": "You are RAGamuffin, a Retrieval-Augmented Generation (RAG) agent, "
                "that can also retrieve text from webpages or YouTube videos when a link is provided. "
                "You will be provided content in one of two different formats. "
                "When RAGing documents, the format will be: 'user_query \n <<document1_name><document1_text>> \n"
                "<<document2_name><document2_text>> ...'. "
                "When querying a webpage or a YouTube video, the format will be: 'user_query \n <webpage_text>'. "
                "Your objective is to generate a response based on the user query and the retrieved document(s), "
                "webpage text, or video transcript."
                }] # Chat history

    while True:

        print("")
        user_input = input(">> ")

        # Exit chat
        if user_input == "/exit":
            print("Bye!")
            break

        # Turn RAG off
        if user_input == "/ragoff":
            rag = False
            print("RAG OFF")
            continue

        # Turn RAG on
        if user_input == "/ragon":
            rag = True
            print("RAG ON")
            continue

        # Change the number of documents to retrieve
        if user_input == "/kdocs":
            k_docs = int(input("Number of documents to retrieve: "))
            print(f"Now the top {k_docs} documents will be RAGed")
            continue

        # Change the minimum similarity score to retrieve a document
        if user_input == "/minscore":
            min_score = float(input("Minimum similarity score to retrieve a document: "))
            print(f"Now the minimum similarity score is {min_score}")
            continue

        # Stop showing the documents retrieved by RAG
        if user_input == "/ragdocsoff":
            rag_docs = False
            print("Name of RAGed documents will not be shown")
            continue

        # Show the documents retrieved by RAG
        if user_input == "/ragdocson":
            rag_docs = True
            print("Name of RAGed documents will be shown")
            continue

        # Specify system prompt
        if user_input == "/system":
            user_input = input("Add system prompt: ")
            history[0] = {
                    "role": "system",
                    "content": user_input,
                    }
            continue

        # Clear the chat history
        if user_input == "/itshistory":
            history = history[0] # Keep the system prompt
            print("Chat history cleared!")
            continue

        # Change LLM
        if user_input == "/changellm":
            llm = input("LLM: ")
            print(f"Now using {llm}")
            continue

        # Chat with the LLM
        else:

            # Query a webpage
            if user_input == "/interwebs":
                url = input("URL: ")
                user_input = input("What do you want to know? >> ")
                web_text = extract_text_from_webpage(url)
                augmented_input = user_input + '\n' + f'<{web_text}>'
                history.append({
                        "role": "user",
                        "content": augmented_input,
                        })
                # Momentarily turn off RAG
                rag_flag = False

            
            if rag and rag_flag:
                
                # Embed the user input and normalize it
                query = ollama.embeddings(model = embedding_model, prompt = user_input)['embedding']

                # Check the query is not empty
                if query:

                    # Process the query
                    query = np.array([query], dtype=np.float32)
                    faiss.normalize_L2(query)

                    # Search for the most similar documents
                    distances, indices = index.search(query, k_docs)

                    # Filter out documents with similarity scores below the threshold
                    indices = indices[0][distances[0] > min_score]

                    # Retrieve the texts of the most similar documents
                    similar_doc_names = [doc_names[i] for i in indices]
                    if rag_docs:
                        print(f'RAGed doc(s): {similar_doc_names}')
                    # If there are similar documents
                    if indices.size > 0:
                        # Concatenate the texts of the most similar documents spacing them with a newline
                        similar_docs_text = '\n'.join([f'<<{doc}>{extract_text_from_txt(doc)}>>' for doc in similar_doc_names])
                        # Concatenate the query and the text of the most similar documents
                        user_input = user_input + '\n' + similar_docs_text

            # If a weebpage was queried, skip this step
            if rag_flag:
                # Append the user input to the history
                history.append({
                    "role": "user",
                    "content": user_input,
                    })
            # Turn on the rag flag in case it was turned off
            rag_flag = True
                
            # Get the response with the LLM
            response = ollama.chat(model = llm, messages = history, stream = True)
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
            print("")

            for chunk in response:
                history.append({
                    "role": "assistant",
                    "content": chunk['message']['content']
                })
                    

if __name__ == "__main__":
    RAGamuffin()