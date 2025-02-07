 # RAGamuffin by Rodrigo González Linares

import ollama
import os
#import PyPDF2 # Yet to be used for PDF extraction
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from googlesearch import search
import itertools
import re
from datetime import date
import json

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
    try:
        # YouTube handling remains the same
        if "youtube.com" in url:
            video_id = get_youtube_id(url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([entry['text'] for entry in transcript])
            return text

        # Modified webpage handling
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Try to find main content
        main_content = None
        content_elements = soup.select('article, [role="main"], main, .content, #content')
        
        if content_elements:
            main_content = content_elements[0].get_text(separator=' ', strip=True)
        elif soup.body:
            # Fallback to body if no content containers found
            main_content = soup.body.get_text(separator=' ', strip=True)
        else:
            # If no body is found, return None
            return None
        
        # Clean up whitespace
        text = ' '.join(main_content.split())
        return text
    except Exception as e:
        print(f"    Warning: Could not extract text from {url}")
        return None

def rag(docu_path,embedding_model):

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

    return index, doc_names

def rag_query(embedding_model, index, doc_names, user_input, k_docs, min_score):

    # Embed the user input and normalize it
    query = ollama.embeddings(model=embedding_model, prompt=user_input)['embedding']

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
        # If there are similar documents
        if indices.size > 0:
            # Concatenate the texts of the most similar documents spacing them with a newline
            similar_docs_text = '\n'.join([f'<<{doc}><{extract_text_from_txt(doc)}>>' for doc in similar_doc_names])
            # Concatenate the query and the text of the most similar documents
            user_input = user_input + '\n' + similar_docs_text

    return similar_doc_names, user_input

def web_search(query, num_web_results):
    # Search for top n Google results
    search_results = search(query, num_results=num_web_results)

    # Extract text from the webpages
    web_page_names = []
    texts = query
    for result in itertools.islice(search_results, num_web_results):
        text = extract_text_from_webpage(result)
        if text:  # Only add pages where text extraction succeeded
            web_page_names.append(result)
            texts += '\n' + f'<<{result}><{text}>>'

    return web_page_names, texts

def list_models(available_models_raw): 

    # Split the string into lines
    lines = available_models_raw.strip().split('\n')

    # Initialize two OrderedDicts to store embedding models and LLMs
    embedding_models = {}
    llms = {}

    # Use a regular expression to extract model names from each line
    embed_count = 1
    llm_count = 1
    for line in lines[1:]:  # Skip the header line
        match = re.match(r'^\s*([\w\-.]+:[\w\-.]+)\s', line)
        if match:
            model_name = match.group(1)
            # Check if the model name contains the word "embed" or "minilm"
            if "embed" in model_name or "minilm" in model_name:
                embedding_models[embed_count] = model_name
                embed_count += 1
            else:
                llms[llm_count] = model_name
                llm_count += 1

    return embedding_models, llms

def select_llm(llms):

    print("Available LLM models:")
    for key, value in llms.items():
        print(f"    {key}: {value}")
    llm_num = input ("Select the LLM model you want to use (type a number):")
    # Check if the user input is a valid key
    while int(llm_num) not in llms.keys():
        llm_num = input("Please select a valid LLM model number: ")
    llm = llms[int(llm_num)]

    return llm

def select_embedding_model(embedding_models):

    print("Available embedding models:")
    for key, value in embedding_models.items():
        print(f"    {key}: {value}")
    embedding_model_num = input ("Select the embedding model you want to use (type a number):")
    # Check if the user input is a valid key
    while int(embedding_model_num) not in embedding_models.keys():
        embedding_model_num = input("Please select a valid embedding model number: ")
    embedding_model = embedding_models[int(embedding_model_num)]

    return embedding_model  

def select_path():
    
        # Get the user to input the path to the documents
        default_docu_path = input("Use default path (docs/)? ([y]/n): ")
        if default_docu_path != "n":
            docu_path = "docs/"
        else:
            docu_path = input("Enter the path to the documents: ")
    
        return docu_path

def select_models_and_path(embedding_models, llms):
        
        # Get the user to select the LLM model
        llm = select_llm(llms)
        # Get the user to select the embedding model
        embedding_model = select_embedding_model(embedding_models)
        # Get the user to input the path to the documents
        docu_path = select_path()
    
        return llm, embedding_model, docu_path

def RAGamuffin():

    # Start Ollama and get available models
    with os.popen('ollama list') as stream:
        available_models = stream.read()

    # Obtain dictionaries with the available embedding models and LLMs
    embedding_models, llms = list_models(available_models)

    # Check if a configuration file exists
    if os.path.exists("config.json"):
        with open("config.json", 'r') as file:
            config = json.load(file)
        # Check if the configuration file is active
        if config["Active"] == True:
            # Check if the models and path in the configuration file are valid
            if config["LLM"] in llms.values() and config["Embedding"] in embedding_models.values():
                llm = config["LLM"]
                routing_llm = config["Routing_LLM"]
                embedding_model = config["Embedding"]
                docu_path = config["Path"]
            # If the models and path in the configuration file are not valid, ask the user for the necessary information
            else:
                llm, embedding_model, docu_path = select_models_and_path(embedding_models, llms)
                routing_llm = llm
        # If the configuration file is not active, ask the user for the necessary information
        else:
            llm, embedding_model, docu_path = select_models_and_path(embedding_models, llms)
            routing_llm = llm
    # If the configuration file does not exist or is not valid, ask the user for the necessary information
    else: 
        llm, embedding_model, docu_path = select_models_and_path(embedding_models, llms)
        routing_llm = llm

    # Get the user to select the mode
    mode = input("Do you want to start the conversation in RAG, web search or conversational mode? ([auto]/rag/web/conv): ")

    # Index the documents if the mode is not web or conversational (i.e., RAG mode)
    indexing_done = False
    if mode != "web" and mode != "conv":
        if mode != "rag":
            auto_mode = True
        else:
            auto_mode = False
        mode = "rag"
        index, doc_names = rag(docu_path,embedding_model)
        indexing_done = True

    # Initialize parameters
    k_docs = 1 # Number of documents to retrieve
    min_score = 0.0 # Minimum similarity score to retrieve a document, orthogonal by default
    rag_docs = True # Show the documents retrieved by RAG if True
    num_web_results = 3 # Number of web results to retrieve
    web_docs = True # Show the webpages retrieved by web search if True
    current_date = date.today() # Get the current date
    formatted_date = current_date.strftime("%Y-%m-%d") # Format the date
    hide_thinking = True # Hide the thinking part of the response
    hide_routing = True # Hide the routing part of the response
    already_ragged = False # Flag to check if a query has already been RAGed
    min_score_auto = 0.6 # Minimum similarity score overide the router with RAG in auto mode

    # Hard-coded system prompt for auto mode router
    auto_hystory = [{
                "role": "system",
                "content": 
                "You are a prompt router agent part of a Retrieval-Augmented Generation (RAG) system called RAGamuffin, that can help the user decide whether to use RAG, web search or simply respond to a prompt. "
                "Moreover, you are capable of reformulating the user query to better suit the use case. "
                "Is the following prompt likely referring to a regular query, one that requires RAG, or one that requires to search the web? "
                "Respond using this format without including any extra text: '<type><reformulated prompt>'. Type can be 'RAG' or 'web'. "
                f"In the case of a normal conversational query, simply do not respond. Today is {formatted_date}. \n"
                "Examples: \n"
                "    Query: 'What's the price of Bitcoin today?' "
                f"    Response: '<web><price of Bitcoin {formatted_date}>' \n"
                "    Query: 'Search through my documents and tell me how to prepare a cheesecake.' "
                "    Response: '<RAG><Cheesecake>' \n"
                "    Query: 'What is RAG?'"
                "    Response: '' \n"
                "It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, "
                "using this information to augment information drawn from its own vast, static training data. This allows LLMs to use domain-specific and/or updated information' \n"
                "VERY IMPORTANT: Always do this for each new query! For example, if the user asks 'which are the main financial news of today?' you will respond "
                f"'<web><financial news {formatted_date}>'. If next the user asks 'and political ones?', you will take into account the prior interaction to know "
                f"he is also talking about news, and respond '<web><political news {formatted_date}>'. Of course, take into account that just because a prior question required a web search, "
                "for example, the next one does not necessarly will require it. It might just be a normal conversational query, for instance. To this end you will have access to past responses here: \n"
                }] 

    # Initialize chat history with the system prompt
    history = [{
                "role": "system",
                "content": "You are RAGamuffin, a Retrieval-Augmented Generation (RAG) agent, "
                "that can also search the web, or retrieve text from specific webpages or YouTube videos when a link is provided. "
                "You will be provided content in the following format: \n"
                "'user_query \n <<document1_name><document1_text>> \n"
                "<<document2_name><document2_text>> ...'. "
                "Your objective is to generate a response based on the user query and the retrieved document(s), "
                "webpage text, or video transcript. If no such resources are provided, you will simply hold a conversation based"
                f"on the chat history. Today is {formatted_date}."
                }] 

    # Main loop
    while True:

        print("")
        user_input = input(">> ")

        # MAGIC WORDS

        # Exit chat
        if user_input == "/exit" or user_input == "/bye":
            print("Bye!")
            break

        # Turn auto mode on
        elif user_input == "/auto":
            auto_mode = True
            print("Auto mode ON")
            continue

        # Turn RAG on
        elif user_input == "/rag":
            mode = "rag"
            auto_mode = False
            print("RAG ON")
            if indexing_done == False:
                index, doc_names = rag(docu_path,embedding_model)
                indexing_done = True
            continue

        # Turn web search on
        elif user_input == "/web":
            mode = "web"
            auto_mode = False
            print("Web search ON")
            continue

        # Turn conversational mode on
        elif user_input == "/conv":
            mode = "conv"
            auto_mode = False
            print("Conversational mode ON")
            continue

        # Change the number of documents to retrieve
        elif user_input == "/kdocs":
            k_docs = int(input("Number of documents to retrieve: "))
            print(f"Now the top {k_docs} documents will be RAGed")
            continue

        # Change the minimum similarity score to retrieve a document
        elif user_input == "/minscore":
            min_score = float(input("Minimum similarity score to retrieve a document: "))
            print(f"Now the minimum similarity score is {min_score}")
            continue

        # Stop showing the documents retrieved by RAG
        elif user_input == "/ragdocsoff":
            rag_docs = False
            print("Name of RAGed documents will not be shown")
            continue

        # Show the documents retrieved by RAG
        elif user_input == "/ragdocson":
            rag_docs = True
            print("Name of RAGed documents will be shown")
            continue

        # Change the number of web results to retrieve
        elif user_input == "/kweb":
            num_web_results = int(input("Number of web results to retrieve: "))
            print(f"Now the top {num_web_results} web results will be used")
            continue

        # Stop showing the webpages retrieved by web search
        elif user_input == "/webdocsoff":
            web_docs = False
            print("Webpages retrieved by web search will not be shown")
            continue

        # Show the webpages retrieved by web search
        elif user_input == "/webdocson":
            web_docs = True
            print("Webpages retrieved by web search will be shown")
            continue

        # Specify system prompt
        elif user_input == "/system":
            user_input = input("Add system prompt: ")
            history[0] = {
                    "role": "system",
                    "content": user_input,
                    }
            continue

        # Clear the chat history
        elif user_input == "/itshistory":
            history = [history[0]] # Keep the system prompt
            print("Chat history cleared!")
            continue

        # Change LLM
        elif user_input == "/changellm":
            llm = select_llm(llms)
            continue

        # Query a webpage
        elif user_input == "/interwebs":
            url = input("URL: ")
            user_input = input("What do you want to know? >> ")
            web_text = extract_text_from_webpage(url)
            user_input = user_input + '\n' + f'<<{url}><{web_text}>>'

        # Toggle off showing the thinking section
        elif user_input == "/thinkhide":
            hide_thinking = True
            print("Thinking section will be hidden")
            continue

        # Toggle on showing the thinking section
        elif user_input == "/thinkshow":
            hide_thinking = False
            print("Thinking section will be shown")

        # Toggle off showing the routing section
        elif user_input == "/routehide":
            hide_routing = True
            print("Routing section will be hidden")
            continue

        # Toggle on showing the routing section
        elif user_input == "/routeshow":
            hide_routing = False
            print("Routing section will be shown")

        # Change the minimum similarity score to retrieve a document in auto mode
        elif user_input == "/minscoreauto":
            min_score_auto = float(input("Minimum similarity score to retrieve a document in auto mode: "))
            print(f"Now the minimum similarity score in auto mode is {min_score_auto}")
            continue

        # Change the routing LLM
        elif user_input == "/changeroutingllm":
            routing_llm = select_llm(llms)
            continue

        # List magic words
        elif user_input == "/magicwords" or user_input.startswith("/"):
            if user_input == "/magicwords":
                print("Magic words:")
            else:
                print("Invalid magic word. Valid magic words:")
            print("    /exit or /bye: Quit the chat")
            print("    /auto: Activate auto mode")
            print("    /rag: Activate RAG mode")
            print("    /web: Activate web search mode")
            print("    /conv: Activate conversation-only mode")
            print("    /interwebs: Provide a URL to a webpage or YouTube video and ask questions about it")
            print("    /itshistory: Clear the chat history")
            print("    /changellm: Change the LLM model on the fly while preserving the chat history! Allows you to use the best model to handle the specific task at hand!")
            print("    /kdocs: Change the number of documents to be retrieved for RAG (1 by default)")
            print("    /minscore: Change the minimum cosine similarity score (from -1.0 for most dissimilar to 1.0 for most similar) to retrieve a document (0.0 by default)")
            print("    /ragdocsoff: Disable printing the names of the documents used for RAG")
            print("    /ragdocson: Enable printing the names of the documents used for RAG (shown by default)")
            print("    /kweb: Change the number of web pages to be retrieved during web search (3 by default)")
            print("    /webdocsoff: Disable printing the names of the web pages used for web search")
            print("    /webdocson: Enable printing the names of the web pages used for web search (shown by default)")
            print("    /system: Provide a system prompt to change the behaviour of the LLM (e.g., 'When reviewing code, explain what each function does thoroughly, yet in simple terms.')")
            print("    /thinkhide: Hide the thinking section in the response (hidden by default)")
            print("    /thinkshow: Show the thinking section in the response")
            print("    /routehide: Hide the routing section in the response (hidden by default)")
            print("    /routeshow: Show the routing section in the response")
            print("    /minscoreauto: Change the minimum similarity score to retrieve a document in auto mode (0.6 by default)")
            print("    /changeroutingllm: Change the routing LLM model on the fly while preserving the chat history! (Same as main LLM by default)")
            continue

        # Chat with the LLM
        else:

            if auto_mode == True:

                # Overwrite with RAG if a query matches a document
                similar_doc_names, user_input = rag_query(embedding_model, index, doc_names, user_input, k_docs, min_score_auto)

                if similar_doc_names != []:
                    mode = "rag"
                    already_ragged = True

                else:

                    # Append the user input to the history
                    auto_hystory.append({
                            "role": "user",
                            "content": user_input,
                            })

                    # Ask LLM to choose the mode
                    auto_response = ollama.chat(model = routing_llm, messages = auto_hystory, stream = True)

                    # Print a routing message if the routing section is hidden
                    if hide_routing == True:
                        print("    Routing ...")

                    # Print the response and concatenate the chunks
                    response_content = ""
                    for chunk in auto_response:
                        chunk_content = chunk['message']['content']                    
                        if hide_thinking and chunk_content == "<think>":
                            if hide_routing == False:
                                print("    Thinking ...")
                        elif hide_thinking == False or ("</think>" in response_content and chunk != "</think>") or "r1" not in routing_llm: # R1 is the only reasoning model in Ollama so far
                            if hide_routing == False:
                                print(chunk_content, end='', flush=True)
                        response_content += chunk_content
                    print("")

                    # Eliminate the last user input
                    auto_hystory.pop()

                    # Remove the thinking section in case it is present
                    response_content = re.sub(r'<think>.*?</think>', '', response_content)

                    # Extract the mode from the response
                    if "<RAG>" in response_content:
                        mode = "rag"
                        extracted_query = re.search(r'<RAG><([^>]*)>', response_content)
                        extracted_query = extracted_query.group(1) if extracted_query else user_input
                        print("\n    EXTRACTED QUERY: " + extracted_query + "\n")
                        # Append both query and response to system prompt for future reference
                        auto_hystory[0]["content"] = auto_hystory[0]["content"] + f"    Query: '{user_input}' Response: '<RAG><{extracted_query}>' \n"
                    elif "<web>" in response_content:
                        mode = "web"
                        extracted_query = re.search(r'<web><([^>]*)>', response_content)
                        extracted_query = extracted_query.group(1) if extracted_query else user_input
                        print("\n    EXTRACTED QUERY: " + extracted_query + "\n")
                        # Append both query and response to system prompt for future reference
                        auto_hystory[0]["content"] = auto_hystory[0]["content"] + f"    Query: '{user_input}' Response: '<web><{extracted_query}>' \n"
                    else:
                        mode = "conv"
                        # Append query without response for conversational mode
                        auto_hystory[0]["content"] = auto_hystory[0]["content"] + f"    Query: '{user_input}' Response: '' \n"

            # Perform RAG if the RAG mode is on
            if mode == "rag":
                
                # Perform the RAG query
                if auto_mode == False:
                    similar_doc_names, user_input = rag_query(embedding_model, index, doc_names, user_input, k_docs, min_score)
                elif already_ragged == True:
                    already_ragged = False
                else:
                    similar_doc_names, user_input = rag_query(embedding_model, index, doc_names, extracted_query, k_docs, min_score)


                # Show the documents retrieved if the flag is on
                if rag_docs:
                    print(f'    RAGed doc(s): {similar_doc_names}')
                    print("")

            # Query the web if the web search mode is on
            elif mode == "web":

                # Search the web
                if auto_mode == False:
                    web_page_names, user_input = web_search(user_input, num_web_results)
                else:
                    web_page_names, user_input = web_search(extracted_query, num_web_results)

                # Show the webpages retrieved if the flag is on
                if web_docs:
                    print(f'    Googled webpage(s): {web_page_names}')
                    print("")

            # Append the user input to the history
            history.append({
                    "role": "user",
                    "content": user_input,
                    })
                
            #if auto_mode == False or (auto_mode == True and mode != "conv"):

            # Get the response with the LLM
            response = ollama.chat(model = llm, messages = history, stream = True)

            # Print the response and concatenate the chunks
            response_content = ""
            for chunk in response:
                chunk_content = chunk['message']['content']                    
                if hide_thinking and chunk_content == "<think>":
                    print("    Thinking ...")
                elif hide_thinking == False or ("</think>" in response_content and chunk != "</think>") or "r1" not in llm: # R1 is the only reasoning model in Ollama so far
                    print(chunk_content, end='', flush=True)
                response_content += chunk_content
            print("")

            # Append the entire response as a single message
            history.append({
                "role": "assistant",
                "content": response_content
            })

# Main function
if __name__ == "__main__":

    # Start RAGamuffin
    RAGamuffin()