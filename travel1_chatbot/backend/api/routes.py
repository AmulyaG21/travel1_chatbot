from fastapi import APIRouter, UploadFile, Form, Request, HTTPException
from services.file_processing.file_handler import FileHandler
from services.rag_pipeline import RAGPipeline
from core.database import QdrantDB
from fastapi.responses import StreamingResponse
from api.travel_api import TravelAPIClient
import os
import re

router = APIRouter()
file_handler = FileHandler()
rag_pipeline = RAGPipeline()
travel_client = TravelAPIClient()
_db = QdrantDB()

@router.post("/flights/query")
def search_flights_query(query: str):
    """
    Example query: "flights from JFK to LHR from 2025-10-15 to 2025-10-22"
    """
    pattern = r"flights from (\w+) to (\w+) from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})"
    match = re.search(pattern, query, re.IGNORECASE)

    if not match:
        return {
            "error": "Query format not understood. Please use: 'flights from ORIGIN to DEST from YYYY-MM-DD to YYYY-MM-DD'"
        }

    origin, destination, outbound_date, return_date = match.groups()

    response = travel_client.search_flights(
        origin=origin,
        destination=destination,
        outbound_date=outbound_date,
        return_date=return_date
    )

    outbound_flights, return_flights = [], []

    for option in response.get("best_flights", []):
        for flight in option.get("flights", []):
            flight_info = {
                "airline": flight.get("airline"),
                "flight_number": flight.get("flight_number"),
                "departure": flight.get("departure_airport", {}).get("name"),
                "departure_time": flight.get("departure_airport", {}).get("time"),
                "arrival": flight.get("arrival_airport", {}).get("name"),
                "arrival_time": flight.get("arrival_airport", {}).get("time"),
                "duration_minutes": flight.get("duration"),
                "travel_class": flight.get("travel_class"),
            }

            if flight.get("departure_airport", {}).get("id") == origin:
                outbound_flights.append(flight_info)
            else:
                return_flights.append(flight_info)

    return {
        "origin": origin,
        "destination": destination,
        "outbound_date": outbound_date,
        "return_date": return_date,
        "results": {
            "outbound": outbound_flights,
            "return": return_flights
        }
    }

@router.post("/upload-file")
async def upload_file(file: UploadFile):
    """Accept a single file, ingest it, and return stored chunk count."""
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
    request: Request,
    query: str = Form(None),
    stream: bool = Form(False),
    
):
    """Handle file-based queries only"""
    # If no form data, try to get JSON
    if query is None:
        try:
            data = await request.json()
            query = data.get('query', '')
            stream = data.get('stream', False)
            
        except:
            raise HTTPException(status_code=400, detail="Invalid request format. Send as form or JSON with 'query' field")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # Directly handle the query with RAG pipeline
        if stream:
            generator = rag_pipeline.stream_answer(query=query, conversation_id=None)
            return StreamingResponse(generator, media_type="text/plain")
        else:
            result = rag_pipeline.answer_query(query=query, conversation_id=None)
            # return {"response": result, "collection_name": "DocumentQuery"}
            if isinstance(result, dict):
                return {
                    "response": result.get("response", "No response generated"),
                    "citations": result.get("citations", [])
                }
            return {"response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/clear-db")
async def clear_db():
    """Danger: deletes and recreates the default collection with the expected schema."""
    _db.clear_collection()
    return {"status": "ok", "message": "Collection cleared and recreated."}