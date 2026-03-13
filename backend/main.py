"""
Main FastAPI Application — Multi-Agentic RAG: College Study Assistant
"""
import os
import json
import tempfile
import traceback
from io import BytesIO
from typing import List
from contextlib import asynccontextmanager
from  dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from vector_store import get_embedding_model, build_retriever
from agents import create_tools
from graph import build_graph

load_dotenv()  # Load environment variables from .env file
# ------------------- API Keys -------------------
# Load from environment variables - set these in your .env file
required_keys = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID", "GROQ_API_KEY"]
for key in required_keys:
    if key not in os.environ:
        print(f"⚠️ Warning: {key} not found in environment variables")

# ------------------- Global State -------------------
embedding_model = None
retriever = None
uploaded_file_info: List[dict] = []  # list of {"path": ..., "name": ...}

# Available models
AVAILABLE_MODELS = {
    "gpt-oss-120b": {
        "display_name": "GPT-OSS 120B",
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
    },
    "gpt-oss-20b": {
        "display_name": "GPT-OSS 20B",
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
    },
    "kimi-k2": {
        "display_name": "Kimi K2",
        "provider": "groq",
        "model_id": "moonshotai/kimi-k2-instruct-0905",
    },
}


def get_llm(model_key: str):
    """Get the LLM instance for the given model key."""
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")

    info = AVAILABLE_MODELS[model_key]
    if info["provider"] == "groq":
        return ChatGroq(
            model_name=info["model_id"],
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
            max_tokens=4000,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {info['provider']}")


# ------------------- Lifespan (replaces deprecated on_event) -------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global embedding_model
    try:
        embedding_model = get_embedding_model(os.environ["GEMINI_API_KEY"])
        print("✅ Embedding model initialized successfully.")
    except Exception as e:
        print(f"⚠️ Embedding model initialization failed: {e}")
        print("   Embedding model will be initialized on first upload.")
        embedding_model = None
    yield
    # Cleanup (if needed) on shutdown
    print("🛑 Shutting down Multi-Agentic RAG API.")


# ------------------- FastAPI App -------------------
app = FastAPI(title="Multi-Agentic RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------- Pydantic Models -------------------
class QueryRequest(BaseModel):
    query: str
    model_name: str = "gpt-oss-120b"


class ChatRequest(BaseModel):
    query: str
    model_name: str = "gpt-oss-120b"


class TTSRequest(BaseModel):
    text: str


# ------------------- Endpoints -------------------
@app.get("/api/models")
async def get_models():
    """Return available LLM models."""
    return {
        "models": [
            {"key": k, "display_name": v["display_name"]}
            for k, v in AVAILABLE_MODELS.items()
        ]
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more PDF files and build the vector store."""
    global retriever, uploaded_file_info, embedding_model

    if embedding_model is None:
        try:
            embedding_model = get_embedding_model(os.environ["GEMINI_API_KEY"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding model not available: {e}")

    new_files = []
    for upload_file in files:
        if not upload_file.filename.lower().endswith(".pdf"):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tmp:
            content = await upload_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        file_info = {"path": tmp_path, "name": upload_file.filename}
        uploaded_file_info.append(file_info)
        new_files.append(upload_file.filename)

    if not new_files:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")

    try:
        retriever = build_retriever(uploaded_file_info, embedding_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building vector store: {e}")

    return {
        "message": f"Successfully processed {len(new_files)} file(s).",
        "files": new_files,
        "total_files": len(uploaded_file_info),
    }


@app.get("/api/files")
async def get_files():
    """Return list of uploaded files."""
    return {
        "files": [f["name"] for f in uploaded_file_info]
    }


def _extract_response(output: dict):
    """
    Walk the message list backwards to find the final ToolMessage.
    Handle the case where search_agent returns a JSON dict (serialized as string by ToolNode),
    and other tools return plain strings.
    """
    final_response = None
    sources = []
    agent_name = output.get("next_tool", "unknown")
    subtool_name = output.get("subtool", "none")

    for msg in reversed(output["messages"]):
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content

        # ToolNode serializes dict returns as JSON strings
        if isinstance(content, str):
            # Try to parse as JSON (search_agent output)
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    sources = parsed.get("sources", [])
                    final_response = parsed.get("content", content)
                    break
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            # Plain string from other tools
            final_response = content
        elif isinstance(content, dict):
            sources = content.get("sources", [])
            final_response = content.get("content", str(content))
        else:
            final_response = str(content)
        break

    return {
        "response": final_response or "No response generated.",
        "agent": agent_name,
        "sub_agent": subtool_name,
        "sources": sources,
    }


@app.post("/api/query")
async def process_query(request: QueryRequest):
    """Process a query through the multi-agent RAG pipeline."""
    global retriever

    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload PDFs first."
        )

    try:
        llm = get_llm(request.model_name)
        tools = create_tools(retriever, llm)
        agent_app = build_graph(tools, llm)

        state = {
            "messages": [HumanMessage(content=request.query)],
            "next_tool": "",
            "subtool": "",
        }
        output = agent_app.invoke(state)
        return _extract_response(output)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Free-form chat with the LLM (no RAG context)."""
    try:
        llm = get_llm(request.model_name)
        prompt = (
            f"You are a friendly, knowledgeable AI study assistant for college students.\n"
            f"Respond helpfully, concisely, and in a supportive tone.\n\n"
            f"User: {request.query}\n\nAssistant:"
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        return {
            "response": content,
            "agent": "chat_agent",
            "sub_agent": "none",
            "sources": [],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech audio (MP3)."""
    try:
        from gtts import gTTS

        # Limit text length to avoid very long audio generation
        tts_text = request.text
        if len(tts_text) > 5000:
            tts_text = tts_text[:5000]
        tts = gTTS(text=tts_text, lang="en", slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"},
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="gTTS not installed. Run: pip install gTTS"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.delete("/api/files")
async def clear_files():
    """Clear all uploaded files and reset the vector store."""
    global retriever, uploaded_file_info
    retriever = None
    uploaded_file_info = []
    return {"message": "All files cleared."}
