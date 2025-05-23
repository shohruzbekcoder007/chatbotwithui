from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from models.admin import Admin
from auth.admin_auth import get_current_admin

router = APIRouter(tags=["admin"])

@router.get("/admin")
async def admin_page():
    return FileResponse('static/admin.html')
