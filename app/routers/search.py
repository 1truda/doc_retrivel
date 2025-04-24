from fastapi import APIRouter


router = APIRouter()

@router.get("/")
async def search_documents(query: str):
    return {"results": f"Searching for {query}"}