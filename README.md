<div align="center">
<img src="https://github.com/RGonzLin/RAGamuffin/assets/65770155/caf10cbb-15aa-42d7-b4db-085265109999" alt="RAGamuffin" width="500">

  **A minimalistic and versatile RAG implementation built on top of Ollama, that can also search the web.**

</div>

## Key features
* Load your favourite LLM and embedding model from Ollama.
* With the power of Retrieval-Augmented Generation (RAG), ask questions to the LLM about your documents... **LOCALLY!**
* **NEW!** With a simple prompt search the web (via Google) and let the LLM answer all your questions Perplexity-style.
* Toggle between RAG, web search and conversation modes.
* Provide a link to a webpage or YouTube video and query the LLM regarding it.
* Change LLMs on the fly without clearing the conversation history.
* **Now RAGamuffin is more capable with the addition of `llama3.1` to Ollama, try it out!**

## Get started 
1. Download Ollama: `https://ollama.com`.
2. `pip install ollama PyPDF2 faiss numpy requests beautifulsoup4 youtube_transcript_api googlesearch-python`.
3. Download some models by typing `ollama pull` followed by the name of the model, in the terminal (I recommend `llama3.1` for the LLM and `mxbai-embed-large` for the embedding model).
4. Run the script by typing `python ragamuffin.py`.
5. All done! Start chatting about your documents with complete privacy.

After this first setup, you will only need to **start from step 4**.  

## In the chat
Simply type to chat with the LLM!  

There are some magic words however. All of them start with a `/`, and allow you to perform special actions:
* `/rag`: Activate RAG mode (ON by default).
* `/web`: Activate web search mode.
* `/conv`: Activate conversation-only mode.
* `/interwebs`: Provide a URL to a webpage or YouTube video and ask questions about it. 
* `/itshistory`: Clear the chat history.
* `/changellm`: Change the LLM model on the fly while preserving the chat history! Allows you to use the best model to handle the specific task at hand!
* `/exit`: Quit the chat.
* `/kdocs`: Change the number of documents to be retrieved for RAG (1 by default).
* `/minscore`: Change the minimum cosine similarity score (from -1.0 for most dissimilar to 1.0 for most similar) to retrieve a document (0.0 by default).
* `/ragdocsoff`: Disable printing the names of the documents used for RAG.
* `/ragdocson`: Enable printing the names of the documents used for RAG (shown by default).
* `/kweb`: Change the number of web pages to be retrieved during web search (3 by default).
* `/webdocsoff`: Disable printing the names of the web pages used for web search.
* `/webdocson`: Enable printing the names of the web pages used for web search (shown by default).
* `/system`: Provide a system prompt to change the behaviour of the LLM (e.g., "When reviewing code, explain what each function does thoroughly, yet in simple terms."). **†**
* `/magicwords`: List all the magic words. 

## The models
**You can set up default LLM and embedding models from the `config.json` file**, and setting `Active` to `true`. `llama3.1` (8B) for the LLM, and `all-minilm` (23M) for the embedding model are generally good choices, but you might consider some other options depending on your specific needs:
* You have many documents and `all-minilm` is not cutting it? Try `mxbai-embed-large` (334M).
* Your documents are too big and `llama3` simply does not have a big enough context window (8K tokens)? Use `llama3-gradient`; also 8M parameters, but with a context window of over 1M!
* `llama3` is too dumb? Go for `llama3:70b` if your computer can handle it!
* Need a tiny model due to hardware constraints? Give `phi3:mini` (3B) a chance.  
* Too European to run one of them darn American models? `mistral` (7B) is for you!

Be mindful that smaller models, even when documents are able to fit within the context window, might not "remember" long-term information very well, nor perform adequately in needle-in-a-haystack-like tasks, compared to more capable models like GPT-4.  

If you would rather specify the models (and the path to the documents for RAG) each time you start RAGamuffin, set `Active` in the configuration file to `false`. 

## Some prompts to try
The `docs` folder contains some sample documents; a Markdown with a couple of recipes, a Python file with an implementation of the game Snake (and yes, you can play it!), and the RAGamuffin file itself. Ask anything relating to these documents... or a webpage... or a YouTube video.   

Here are some ideas to get you started:

### When in RAG mode
* `>> How can I bake a cheesecake?`
* `>> When does a game of snake ends?`
* `>> What does /changellm do regarding RAGamuffin?`

### When in web search mode
* `>> What are the most important news stories of today?`

### When providing a specific URL
* `>> /interwebs`  
 `URL: https://www.quantamagazine.org/game-theory-can-make-ai-more-correct-and-efficient-20240509/`  
 `What do you want to know? >> Summarize the article`  
* `>> /interwebs`  
 `URL: https://www.youtube.com/watch?v=PtfatBOlHIA`  
 `What do you want to know? >> What is this video about?`

## Notes

### Troubleshooting 
After searching the web for a while under normal use, you might encounter a 'too many requests' HTTP error from Google. This can be easily circumvented by switching IPs using a VPN.

### Upcoming planned features
* An 'Automatic Mode' which intelligently determines the best response method (RAG, web search or conversation) based on the user's query.
* PDF support.
* Include a magic word so that users can add new UTF-8 encoded document extensions to be processed by RAG. Without modifying the Python script, the extensions '.txt', '.md', '.py', '.sh', and '.js' will be processed by RAG in this current version.



**†** The deafault system prompt is:  
"You are RAGamuffin, a Retrieval-Augmented Generation (RAG) agent, that can also search the web, or retrieve text from specific webpages or YouTube videos when a link is provided. You will be provided content in the following format:   
'user_query   
<<document1_name><document1_text>>   
<<document2_name><document2_text>> ...'. Your objective is to generate a response based on the user query and the retrieved document(s), webpage text, or video transcript. If no such resources are provided, you will simply hold a conversation based on the chat history. Today is [Current Date]."
