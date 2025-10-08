from fastapi import APIRouter, UploadFile, Form
from file_handler import FileHandler
from rag_pipeline import RAGPipeline
import os
from backend.database import QdrantDB
from fastapi.responses import StreamingResponse

router = APIRouter()
file_handler = FileHandler()
rag_pipeline = RAGPipeline()

# Local DB handle for operations (used by clear-db)
_db = QdrantDB()

@router.post("/upload-file")
async def upload_file(
    file: UploadFile,
):
    """Accept a single file, ingest it, and return stored chunk count.
    Uses the single default collection defined in config.Settings.QDRANT_COLLECTION.
    """
    os.makedirs("./temp", exist_ok=True)

    file_path = f"./temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = file_handler.process_and_store(file_path)
    return {
        "message": "File uploaded and processed successfully",
        **result,
        "filename": file.filename,
    }


@router.post("/query")
async def query_text(
    query: str = Form(...),
    stream: bool = Form(default=False),
):
    """Answer a query using the default collection. Conversation tracking is handled internally.
    - stream: if true, stream the LLM output tokens as they arrive
    """
    if stream:
        generator = rag_pipeline.stream_answer(query=query, conversation_id=None)
        return StreamingResponse(generator, media_type="text/plain")
    else:
        result = rag_pipeline.answer_query(query=query, conversation_id=None)
        return {"response": result}


@router.post("/clear-db")
async def clear_db():
    """Danger: deletes and recreates the default collection with the expected schema."""
    _db.clear_collection()
    return {"status": "ok", "message": "Collection cleared and recreated."}
