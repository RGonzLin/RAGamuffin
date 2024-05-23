<div align="center">
<img src="https://github.com/RGonzLin/RAGgamuffin/assets/65770155/efb9d019-6971-4f88-8547-6463e8e5f8fe" alt="RAGgamuffin" width="500">

  **A minimalistic and versatile RAG implementation built on top of Ollama, with a couple of tricks up its sleeve.**

</div>

## Key features
* Load your favourite LLM and embedding model from Ollama.
* With the power of Retrieval-Augmented Generation (RAG), ask questions to the LLM about your documents... **LOCALLY!**
* Toggle RAG on and off with a simple command.
* Provide a link to a webpage or YouTube video and query the LLM regarding it.

## Get started 
1. Download Ollama: `https://ollama.com`.
2. `pip install PyPDF2 faiss requests beautifulsoup4 youtube_transcript_api`.
3. Start Ollama by clicking the App icon, this will not open any window but it will run on the background.
4. Download some models by typing `ollama pull` followed by the name of the model, in the terminal (I recommend `llama3` for the LLM and `all-minilm` for the embedding model.)
5. Run the script by typing `python raggamuffin.py`.
6. Choose the default models and documents directory or specify your own.
7. All done! start chatting about your documents with complete privacy.

After this first setup, you will only need to start at step 5!

## In the chat
Simply type to chat with the LLM!  

There are some magic words however, all of them start with a `/`, and allow you to perform special actions:
* `/interwebs`: Provide a URL to a webpage or YouTube video and ask questions about it. 
* `/system`: Provide a system prompt to change the behaviour of the LLM (i.e., "When reviewing code, explain what each function does thoroughly, yet in simple terms."
* `/exit`: Quit the chat.
* `/ragoff`: Disable RAG capabilities; for chatting about previously retrived documents or just having a normal conversation (ON by default).
* `/ragon`: Enable RAG capabilities.
* `/kdocs`: Change the number of documents to be retrieved for RAG (1 by default).
* `/minscore`: Change the minimum cosine similarity (from -1 to most dissimilar to 1 for most similar) score to retrieve a document (0.0 by default).
* `/ragdocsoff`: Disable printing the names of the documents used for RAG (ON by default).
* `/ragdocson`: Enable printing the names of the documents used for RAG.

## The models
As previously mentioned, the default models are `llama3`(8B) for the LLM, and `all-minilm` (23M) for the embedding model. This are generally good choices, but you might consider some other models depending on your specific needs:
* You have many documents and `all-minilm` is not cutting it? try `mxbai-embed-large` (334M).
* Your documents are too big and `llama3` simply does not have a big enough context window (8K window)? Use `llama3-gradient`; also 8M parameters, but with a context window of over 1M!
* `llama3` is too dumb? Go for `llama3:70b` if your computer can handle it!
