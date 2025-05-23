from fastapi import APIRouter, Request, Depends
from models.feedback import Feedback
from models.admin import Admin
from auth.admin_auth import get_current_admin

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
async def get_feedbacks(admin: Admin = Depends(get_current_admin)):
    try:
        feedbacks = await Feedback.find_all().to_list()
        return {"success": True, "feedbacks": feedbacks}
    except Exception as e:
        return {"success": False, "error": str(e)}
