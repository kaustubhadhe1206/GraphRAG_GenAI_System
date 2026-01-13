#!pip install langchain python-dotenv langchain_community arxiv wikipedia pymupdf langchain-text-splitters langchain_huggingface streamlit langchain_groq youtube_transcript_api cassio tiktoken PyPDF2 langgraph langchain_core langchain_classic langchainhub pydantic

import sys
sys.modules["tensorflow"] = None
sys.modules["keras"] = None
sys.modules["tf_keras"] = None

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st

if not hasattr(st.session_state, "messages"):
    st.session_state.messages = None

if not hasattr(st.session_state, "embeddings"):
    st.session_state.embeddings = None

if not hasattr(st.session_state, "llm"):
    st.session_state.llm = None

if not hasattr(st.session_state, "cassio_init"):
    st.session_state.cassio_init = None

if not hasattr(st.session_state, "astra_vector_store"):
    st.session_state.astra_vector_store = None

if not hasattr(st.session_state, "ddg"):
    st.session_state.ddg = None

if not hasattr(st.session_state, "wiki"):
    st.session_state.wiki = None

if not hasattr(st.session_state, "ffmpeg_path"):
    st.session_state.ffmpeg_path = None

if not hasattr(st.session_state, "ingested"):
    st.session_state.ingested = None

if not hasattr(st.session_state, "processing"):
    st.session_state.processing = None

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
ASTRA_DB_TOKEN = st.secrets["ASTRA_DB_TOKEN"]
# ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
# ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def data_ingestion(path_or_url, type_):
    if(type_=="pdf"):
        documents = []

        pdf_loader = PyPDFLoader(path_or_url)
        docs = pdf_loader.load()
        documents.extend(docs) #append documents
        return documents
    elif(type_=="web"):
        web_loader = WebBaseLoader(path_or_url)
        docs = web_loader.load()
        return docs

def chunking(docs, type_):
    if(type_=="pdf"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        docs_split = text_splitter.split_documents(docs)
        return docs_split
    elif(type_=="web"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        docs_split = text_splitter.split_documents(docs)
        return docs_split
    elif(type_=="llm_generated_doc"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        docs_split = text_splitter.split_documents(docs)
        return docs_split

def update_vectorstore(docs_split):
    st.session_state.astra_vector_store.add_documents(docs_split)

def reset_vectorstore():
    st.session_state.astra_vector_store.clear()


from langchain_core.prompts import ChatPromptTemplate

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Decide where to route the question.

Choose ONE:
- wiki_search
- browser_search
- youtube_summarize

Rules:
- Use youtube_summarize if the question contains a YouTube link
- Use browser_search for recent or changing info
- Use wiki_search for general knowledge

Return ONLY the datasource name."""),
    ("human", "{question}")
])

from typing import TypedDict, List

class GraphState(TypedDict):
  question: str
  datasource: str
  documents: List[str]
  answer: str

def vectorstore_check_node(state: GraphState):
    question = state["question"]

    results = st.session_state.astra_vector_store.similarity_search_with_score(
        question, k=1
    )

    if not results:
        return {"datasource": "router"}

    _, score = results[0]
    print("Score-----------------------------------------------")
    print(score)
    if score >= 0.7:
        print("VectorStore-----------------------------------------------")
        return {"datasource": "vectorstore"}
    else:
        print("Router-----------------------------------------------")
        return {"datasource": "router"}


# Router Node
def router_node(state: GraphState):
    response = st.session_state.llm.invoke(
        router_prompt.format_messages(
            question=state["question"]
        )
    )
    print(f"{response.content.strip()}-----------------------------------------------")

    return {"datasource": response.content.strip()}

# VectorStore Node
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant.
Answer the question strictly using the provided context, and give detailed explanation.
If the answer is not present in the context, say "I don't know based on the provided documents."""),
    ("human",
     """Context:
{context}

Question:
{question}""")
])

def vectorstore_node(state: GraphState):
    question = state["question"]
    docs = st.session_state.astra_vector_store.similarity_search_with_score(question,k=3)
    # Combine retrieved chunks to create context
    context = "\n\n".join(str(doc[0]) for doc in docs)

    # Invoking LLM to generate response
    response = st.session_state.llm.invoke(
        rag_prompt.format_messages(
            context=context,
            question=question
        )
    )

    print("--------------------------------------------------------------------------------")
    print(response)
    return {
        "answer": str(response.content),
        "documents": docs
    }


# Wikipedia Search Node
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a factual assistant. Answer ONLY using the provided Wikipedia content. "
        "Generate a verbose response only when asked to."
        "Do not use external knowledge or assumptions."
    ),
    (
        "human",
        "Wikipedia content:\n{wiki_content}\n\n"
        "Question:\n{question}\n\n"
        "If the content is insufficient, say so explicitly."
    )
])

if st.session_state.wiki is None:
    st.session_state.wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3)
    )

def wiki_node(state: GraphState):
    question = state["question"]

    wiki_content = st.session_state.wiki.run(question)

    messages = wiki_prompt.format_messages(
        question=question,
        wiki_content=wiki_content
    )

    response = st.session_state.llm.invoke(messages)

    return {
        "answer": response.content.strip()
    }


# DuckDuckGo search
from langchain_community.tools import DuckDuckGoSearchRun

search_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a web-search-based assistant. Answer the question ONLY using "
        "the provided search results. Do not add external knowledge."
        "you must answer what is asked and not provide any other unnecessary information."
    ),
    (
        "human",
        "Search results:\n{search_content}\n\n"
        "Question:\n{question}\n\n"
        "If the results are insufficient, say so explicitly."
    )
])
search = DuckDuckGoSearchRun()

def browser_node(state: GraphState):
    question = state["question"]

    # Fetch search results
    search_content = search.run(question)

    # Format messages
    messages = search_prompt.format_messages(
        question=question,
        search_content=search_content
    )
    
    response = st.session_state.llm.invoke(messages)

    return {
        "answer": response.content.strip()
    }

# Youtube Summarize Node
import re
import yt_dlp
import assemblyai as aai

def extract_video_id(text: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    raise ValueError("No valid YouTube video ID found")

def download_audio(video_id: str) -> str:
    output_file = f"{video_id}"
    os.environ["FFMPEG_BINARY"] = st.session_state.ffmpeg_path
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_file,
        "quiet": True,
        "ffmpeg_location": st.session_state.ffmpeg_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    print(output_file)
    return output_file

def transcribe_audio(audio_file):
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
    # aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    config = aai.TranscriptionConfig(speech_models=["universal"])
    
    transcript = aai.Transcriber(config=config).transcribe(f"./{audio_file}.mp3")
    
    if transcript.status == "error":
      raise RuntimeError(f"Transcription failed: {transcript.error}")
    
    return transcript.text

youtube_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert assistant that summarizes video transcripts. "
        "Use ONLY the transcript provided."
    ),
    (
        "human",
        "Transcript:\n{transcript}\n\n"
        "User request:\n{question}\n\n"
        "Provide a clear, structured summary."
    )
])

def youtube_node(state: GraphState):
    question = state["question"]

    try:
        video_id = extract_video_id(question)
        audio_path = download_audio(video_id)
        transcript_text = transcribe_audio(audio_path)
    except Exception as e:
        return {"answer": f"YouTube processing failed: {e}"}

    messages = youtube_prompt.format_messages(
        transcript=transcript_text,
        question=question
    )

    response = st.session_state.llm.invoke(messages)

    # remove audio file
    if os.path.exists(f"./{audio_path}.mp3"):
        os.remove(f"./{audio_path}.mp3")

    return {
        "answer": response.content.strip(),
        "documents": transcript_text
    }
 

from langgraph.graph import StateGraph, END

# Caching the graph construction
@st.cache_resource
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("vectorstore_check", vectorstore_check_node)
    graph.add_node("vectorstore", vectorstore_node)
    graph.add_node("router", router_node)
    graph.add_node("wiki", wiki_node)
    graph.add_node("browser", browser_node)
    graph.add_node("youtube", youtube_node)

    graph.set_entry_point("vectorstore_check")

    graph.add_conditional_edges(
        "vectorstore_check",
        lambda s: s["datasource"],
        {
            "vectorstore": "vectorstore",
            "router": "router"
        }
    )

    graph.add_conditional_edges(
        "router",
        lambda s: s["datasource"],
        {
            "wiki_search": "wiki",
            "browser_search": "browser",
            "youtube_summarize": "youtube"
        }
    )

    graph.add_edge("vectorstore", END)
    graph.add_edge("wiki", END)
    graph.add_edge("browser", END)
    graph.add_edge("youtube", END)

    return graph.compile()

app = build_graph()

# Streamlit UI
import streamlit as st

def initialize_heavy_objects():
    if st.session_state.llm is None:
        import cassio
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Cassandra
        import imageio_ffmpeg
        
        # Initialize ffmpeg path
        if st.session_state.ffmpeg_path is None:
            st.session_state.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        # LLM Initialization
        # st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        st.session_state.llm = ChatGroq(groq_api_key=GROQ_API_KEY,model_name="llama-3.1-8b-instant")

        # HuggingFace Embeddings
        if st.session_state.embeddings is None:
            st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name = "all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )

        # Cassandra/AstraDB Initialization
        if st.session_state.cassio_init is None:
            cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
            st.session_state.cassio_init = True

        if st.session_state.astra_vector_store is None:
            st.session_state.astra_vector_store = Cassandra(embedding=st.session_state.embeddings,
                               table_name="pdfquery_db",
                               session=None,
                               keyspace=None)

def main():
    st.title("Multi-Agent RAG Chatbot")

    initialize_heavy_objects()

    # Initialize session flags
    if st.session_state.ingested is None:
        st.session_state.ingested = False
    if st.session_state.processing is None:
        st.session_state.processing = False

    # Optional PDF upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents (optional)", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs"):
        st.session_state.processing = True
        all_docs = []

        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            docs = data_ingestion(temp_path, "pdf")
            all_docs.extend(docs)
            os.remove(temp_path)

        docs_split = chunking(all_docs, "pdf")
        update_vectorstore(docs_split)

        st.session_state.ingested = True
        st.session_state.processing = False
        st.success("PDFs processed successfully!")

    # User query
    user_input = st.text_area("Ask a question")

    if user_input and st.button("Ask"):
        with st.spinner("Thinking..."):
            result = app.invoke({"question": user_input})

        st.subheader("Answer")
        st.write(result.get("answer", "No answer generated"))

        # Show vectorstore documents only if PDFs were ingested
        if st.session_state.ingested and "documents" in result:
            with st.expander("Retrieved Context"):
                st.write(result["documents"])
    
    if st.button("Reset Vector Store"):
        with st.spinner("Clearing vector store..."):
            reset_vectorstore()
            st.session_state.ingested = False
        st.success("Vector store cleared successfully")


if __name__ == "__main__":
    main()