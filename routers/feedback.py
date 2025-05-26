from fastapi import APIRouter, Request, Depends, Query
from models.feedback import Feedback
from models.admin import Admin
from auth.admin_auth import get_current_admin
from typing import Optional

router = APIRouter(prefix="/api", tags=["feedback"])

@router.post("/feedback")
async def submit_feedback(request: Request):
    try:
        data = await request.json()
        feedback = Feedback(
            message_text=data.get("message_text"),
            answer_text=data.get("answer_text"),
            feedback_type=data.get("feedback_type"),
            comment=data.get("comment", ""),
            user_id=data.get("user_id")
        )
        await feedback.insert()
        return {"success": True, "id": str(feedback.id)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/admin/feedbacks")
async def get_feedbacks(
    admin: Admin = Depends(get_current_admin),
    page: Optional[int] = Query(1, ge=1, description="Sahifa raqami"),
    limit: Optional[int] = Query(10, ge=1, le=100, description="Har bir sahifadagi elementlar soni")
):
    try:
        # Umumiy feedback'lar sonini hisoblash
        total_count = await Feedback.count()
        
        # Pagination uchun skip va limit hisoblash
        skip = (page - 1) * limit
        
        # Feedbacklarni olish
        feedbacks = await Feedback.find_all().skip(skip).limit(limit).sort("-created_at").to_list()
        
        # Umumiy sahifalar sonini hisoblash
        total_pages = (total_count + limit - 1) // limit
        
        return {
            "success": True, 
            "feedbacks": feedbacks,
            "pagination": {
                "total": total_count,
                "page": page,
                "limit": limit,
                "total_pages": total_pages
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
