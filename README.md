<div align="center">
<img src="https://github.com/RGonzLin/RAGamuffin/assets/65770155/caf10cbb-15aa-42d7-b4db-085265109999" alt="RAGamuffin" width="500">

  **A minimalistic and versatile RAG implementation built on top of Ollama, with a couple of tricks up its sleeve.**

</div>

## Key features
* Load your favourite LLM and embedding model from Ollama.
* With the power of Retrieval-Augmented Generation (RAG), ask questions to the LLM about your documents... **LOCALLY!**
* Toggle RAG on and off with a simple command.
* Provide a link to a webpage or YouTube video and query the LLM regarding it.
* Change LLMs on the fly without clearing the conversation history. 

## Get started 
1. Download Ollama: `https://ollama.com`.
2. `pip install PyPDF2 faiss requests beautifulsoup4 youtube_transcript_api`.
3. Start Ollama by clicking the App icon, this will not open any window but it will run on the background.
4. Download some models by typing `ollama pull` followed by the name of the model, in the terminal (I recommend `llama3` for the LLM and `all-minilm` for the embedding model).
5. Run the script by typing `python ragamuffin.py`.
6. Choose the default models and documents directory or specify your own.
7. All done! start chatting about your documents with complete privacy.

After this first setup, you will only need to **start Ollama as explained in step 3**, and then go directly to step 5.  


## In the chat
Simply type to chat with the LLM!  

There are some magic words however. All of them start with a `/`, and allow you to perform special actions:
* `/interwebs`: Provide a URL to a webpage or YouTube video and ask questions about it. 
* `/system`: Provide a system prompt to change the behaviour of the LLM (i.e., "When reviewing code, explain what each function does thoroughly, yet in simple terms.").
*  `/itshistory`: Clear the chat history.
*  `/changellm`: Change the LLM model on the fly while preserving the chat history! Allows you to use the best model to handle the specific task at hand!
* `/exit`: Quit the chat.
* `/ragoff`: Disable RAG capabilities; for chatting about previously retrived documents or just having a normal conversation (ON by default).
* `/ragon`: Enable RAG capabilities.
* `/kdocs`: Change the number of documents to be retrieved for RAG (1 by default).
* `/minscore`: Change the minimum cosine similarity score (from -1.0 for most dissimilar to 1.0 for most similar) to retrieve a document (0.0 by default).
* `/ragdocsoff`: Disable printing the names of the documents used for RAG (ON by default).
* `/ragdocson`: Enable printing the names of the documents used for RAG.

An important detail: **Be patient!** Only full responses are shown; they are not streamed word-by-word as you may be used to.

## The models
The default models are `llama3` (8B) for the LLM, and `all-minilm` (23M) for the embedding model. These are generally good choices, but you might consider some other options depending on your specific needs:
* You have many documents and `all-minilm` is not cutting it? Try `mxbai-embed-large` (334M).
* Your documents are too big and `llama3` simply does not have a big enough context window (8K tokens)? Use `llama3-gradient`; also 8M parameters, but with a context window of over 1M!
* `llama3` is too dumb? Go for `llama3:70b` if your computer can handle it!
* Too European to run one of them darn American models? `mistral` (7B) is for you!

## Some prompts to try
The `docs` folder contains some sample documents; a Markdown with a couple of recipes, a Python file with an implementation of the game Snake (and yes, you can play it!), and the RAGamuffin file itself. Ask anything relating to these documents... or a webpage... or a YouTube video.   

Here are some ideas to get you started:

* `>> I am making croquettes, how should I shape them?`
* `>> When does a game of snake ends?`
* `>> What does /changellm do regarding RAGamuffin?`
* `>> /interwebs`  
 `URL: https://www.youtube.com/watch?v=PtfatBOlHIA`  
 `What do you want to know? >> What is this video about?`
* `>> /interwebs`  
 `URL: https://www.theverge.com/2024/5/22/24162429/scarlett-johansson-openai-legal-right-to-publicity-likeness-midler-lawyers`  
 `What do you want to know? >> What is happening between Scarlett Johansson and OpenAI?`

