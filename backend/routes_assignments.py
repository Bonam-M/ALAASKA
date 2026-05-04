from fastapi import APIRouter, Depends, HTTPException
from backend.auth import get_current_user, http_bearer
from fastapi.security import HTTPAuthorizationCredentials
from backend.admin import require_admin, require_grader
from backend.models_assignments import (
    CreateTemplateRequest, 
    CreateAssignmentRequest, 
    UpdateTemplateRequest,
    MarkSolutionRequest,
    SubmitAnswerRequest,
    Question,
    CreateQuizTemplateRequest,
    UpdateQuizTemplateRequest,
    SubmitQuizAnswerRequest,
    UpdateSubmissionSettingsRequest
)
from backend.db_assignments import (
    templates_collection,
    assignments_collection,
    student_assignments_collection,
    quiz_templates_collection,
    student_quiz_responses_collection
)
from backend.db_mongo import conversations_collection, users_collection
from datetime import datetime, timezone
import uuid
from fastapi.responses import StreamingResponse
from backend.pdf_generator import create_gradescope_pdf
from typing import Optional, List

router = APIRouter()

# ========== HELPER FUNCTIONS ==========

def now_utc_iso():
    """Return current UTC time as ISO string for message timestamps"""
    return datetime.now(timezone.utc).isoformat()

def create_message(role: str, content: str) -> dict:
    """
    Create a message with ML analysis metadata.
    Includes timestamp, char_count, word_count for behavioral analysis.
    """
    return {
        "role": role,
        "content": content,
        "timestamp": now_utc_iso(),
        "char_count": len(content),
        "word_count": len(content.split()) if content else 0
    }

def create_assignment_system_prompt(question_number: str, question_text: str, hints: list = None) -> list:
    """
    Create initial messages for an assignment question chat.
    
    Returns list of [system_message, assistant_greeting] with ML metadata
    """
    hints_text = ""
    if hints and len(hints) > 0:
        hints_text = "\n\nAvailable hints for this question:\n" + "\n".join([f"- {hint}" for hint in hints])
    
    system_content = (
        "You are ALAASKA, a supportive teaching assistant. Your job is to guide the user to think critically and find the solution on their own. You DON'T need to tell this to the user."
        "Keep all responses very CONSICE (around 3-4 sentences/points per message). Focus on ONE concept or step at a time. Use Bold or italics for emphasis when helpful."
        "If a student lacks foundational knowledge, provide brief clear explanations, definitions, or short analogies only. "
        "Never reveal full or partial solutions. If the student doesn't understand, ASK them to explain their reasoning first. "
        "Break problems into SMALL steps. After each step, ASK what comes next with a short guiding question. DON't give the next step yourself."
        "If wrong, give ONE simple counterexample or simpler question—never give the fix yourself. "
        "Confirm correctness with minimal explanation (one - two sentences max)."
        "When dealing with math/calculations: explain the approach briefly, don't calculate the final answer. Ask the student to compute and verify their own answer. "
        "If complex topics need multiple steps, SPLIT your responses—don't try to explain everything at once. "
        "Have the student summarize once enough progress is made, then ask them to press 'Mark as Final Answer'. "
        "Understand that you are an AI; if a student reasonably argues your calculation is wrong, re-evaluate graciously. "
        "Assess their level through SHORT guiding questions and use flashcards, mini quizzes, or scenarios when suitable. "
        "Discuss only academic topics."
        "If hints are given, use them to guide the student toward the critical point."
        f"\n\nThe student needs to solve this assignment question:\n\nQuestion {question_number}: {question_text}{hints_text}"
    )
    
    assistant_content = f"""Hi! I'm here to help you work through this assignment question:

**Question {question_number}:** {question_text}

Before we dive in, I'd like to understand your initial thoughts. What's your first impression of this question? What concepts or ideas come to mind when you read it?

Take your time - there's no rush. Let's work through this together! 🎯"""
    
    return [
        create_message("system", system_content),
        create_message("assistant", assistant_content)
    ]

async def check_submission_enabled(assignment_id: str, student_email: str):
    """Check if submissions are enabled for this student"""
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    submissions_enabled = assignment.get("submissions_enabled", True)  # Default: enabled
    submission_exceptions = [e.lower() for e in assignment.get("submission_exceptions", [])]
    
    # If globally enabled, allow everyone
    if submissions_enabled:
        return True
    
    # If globally disabled, check exceptions
    if student_email.lower() in submission_exceptions:
        return True
    
    # Not allowed
    raise HTTPException(
        status_code=403,
        detail="Submissions are currently disabled for this assignment"
    )
# ========== ADMIN ROUTES ==========

@router.post("/assignment-templates")
async def create_assignment_template(
    request: CreateTemplateRequest, 
    user: dict = Depends(require_admin)
):
    """Create a new assignment template (admin only)"""
    template_id = str(uuid.uuid4())
    
    template_doc = {
        "template_id": template_id,
        "title": request.title,
        "description": request.description,
        "questions": [
            {
                "question_id": str(uuid.uuid4()),
                "number": q.number,
                "prompt_md": q.prompt_md,
                "marks": float(q.marks),
                "hints": q.hints or []
            }
            for q in request.questions
        ],
        "created_by": user["email"],
        "created_at": datetime.now(timezone.utc)
    }
    
    await templates_collection.insert_one(template_doc)
    
    return {
        "message": "Template created successfully",
        "template_id": template_id
    }

@router.get("/assignment-templates")
async def get_assignment_templates(user: dict = Depends(require_admin)):
    """Get all assignment templates (admin only)"""
    cursor = templates_collection.find({}, {"_id": 0})
    templates = []
    async for template in cursor:
        templates.append(template)
    return {"templates": templates}

@router.post("/assignments")
async def create_assignment(
    request: CreateAssignmentRequest, 
    user: dict = Depends(require_admin)
):
    """Create a new assignment from a template (admin only)"""
    # Get the template
    template = await templates_collection.find_one({"template_id": request.template_id})
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
   
    if request.pre_quiz_id:
        pre_quiz = await quiz_templates_collection.find_one({"quiz_id": request.pre_quiz_id})
        if not pre_quiz:
            raise HTTPException(status_code=404, detail="Pre-quiz template not found")
    
    if request.post_quiz_id:
        post_quiz = await quiz_templates_collection.find_one({"quiz_id": request.post_quiz_id})
        if not post_quiz:
            raise HTTPException(status_code=404, detail="Post-quiz template not found")
    
    assignment_id = str(uuid.uuid4())
    
    # Create assignment
    assignment_doc = {
        "assignment_id": assignment_id,
        "template_id": request.template_id,
        "title": template["title"],
        "description": template["description"],
        "questions": template["questions"],
        "allowed_students": [email.lower() for email in request.allowed_students],
        "pre_quiz_id": request.pre_quiz_id,  
        "post_quiz_id": request.post_quiz_id,
        "submissions_enabled": True, 
        "submission_exceptions": [], 
        "created_by": user["email"],
        "created_at": datetime.now(timezone.utc)
    }
    
    await assignments_collection.insert_one(assignment_doc)
    
    return {
        "message": "Assignment created successfully",
        "assignment_id": assignment_id
    }
@router.get("/admin/assignments")
async def get_all_assignments(user: dict = Depends(require_admin)):
    """Get all assignments (admin only)"""
    cursor = assignments_collection.find({}, {"_id": 0})
    assignments_list = []
    async for assignment in cursor:
        assignments_list.append({
            "assignment_id": assignment["assignment_id"],
            "template_id": assignment["template_id"],
            "title": assignment["title"],
            "description": assignment["description"],
            "allowed_students": assignment.get("allowed_students", []),
            "created_by": assignment["created_by"],
            "created_at": assignment["created_at"].isoformat()
        })
    return {"assignments": assignments_list}

@router.get("/assignments/{assignment_id}/submission-settings")
async def get_submission_settings(
    assignment_id: str,
    user: dict = Depends(require_admin)
):
    """Get submission settings for an assignment (admin only)"""
    assignment = await assignments_collection.find_one(
        {"assignment_id": assignment_id},
        {"_id": 0, "submissions_enabled": 1, "submission_exceptions": 1}
    )
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    return {
        "submissions_enabled": assignment.get("submissions_enabled", True),
        "submission_exceptions": assignment.get("submission_exceptions", [])
    }

@router.put("/assignments/{assignment_id}/submission-settings")
async def update_submission_settings(
    assignment_id: str,
    request: UpdateSubmissionSettingsRequest,
    user: dict = Depends(require_admin)
):
    """Update submission settings for an assignment (admin only)"""
    submission_exceptions = [email.lower().strip() for email in request.submission_exceptions]
    
    result = await assignments_collection.update_one(
        {"assignment_id": assignment_id},
        {
            "$set": {
                "submissions_enabled": request.submissions_enabled,
                "submission_exceptions": submission_exceptions
            }
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    return {
        "message": "Submission settings updated successfully",
        "submissions_enabled": request.submissions_enabled,
        "submission_exceptions": submission_exceptions
    }

@router.put("/assignments/{assignment_id}/students")
async def update_assignment_students(
    assignment_id: str,
    request: dict,
    user: dict = Depends(require_admin)
):
    """Update the allowed students list for an assignment (admin only)"""
    allowed_students = request.get("allowed_students", [])
    if not allowed_students:
        raise HTTPException(status_code=400, detail="Must provide at least one student email")
    
    # Normalize emails
    allowed_students = [email.lower().strip() for email in allowed_students]
    
    result = await assignments_collection.update_one(
        {"assignment_id": assignment_id},
        {"$set": {"allowed_students": allowed_students}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    return {
        "message": "Assignment updated successfully",
        "assignment_id": assignment_id,
        "total_students": len(allowed_students)
    }




# Update the export_assignment_pdf function

@router.post("/assignments/{assignment_id}/export-pdf")
async def export_assignment_pdf(
    assignment_id: str,
    user: dict = Depends(require_admin)
):
    """Export assignment submissions as Gradescope-compatible PDF (admin only)"""
    try:
        # Get assignment details
        assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        # Get all student submissions
        cursor = student_assignments_collection.find(
            {"assignment_id": assignment_id},
            {"_id": 0}
        )
        
        students_data = []
        async for student_assignment in cursor:
            # Get student details
            user_doc = await users_collection.find_one({"email": student_assignment["student_email"]})
            
            student_name = user_doc.get("username", "Unknown") if user_doc else "Unknown"
            student_email = student_assignment["student_email"]
            
            # ✅ Process questions with submission time
            questions_data = []
            for question in student_assignment["questions"]:
                questions_data.append({
                    "number": question.get("number", "?"),
                    "marks": question.get("marks", 0),
                    "student_solution": question.get("student_solution"),
                    "chat_id": question.get("chat_id"),
                    "submitted_at": question.get("submitted_at")  # ✅ Include submission time
                })
            
            students_data.append({
                "name": student_name,
                "email": student_email,
                "questions": questions_data
            })
        
        if not students_data:
            raise HTTPException(status_code=404, detail="No submissions found for this assignment")
        
        from backend.config import FRONTEND_URL
        base_url = FRONTEND_URL if hasattr(FRONTEND_URL, '__str__') else "http://localhost:3000"
        
        # Generate PDF with base_url
        pdf_buffer = create_gradescope_pdf(
            assignment_title=assignment["title"],
            students_data=students_data,
            base_url=base_url
        )
        
        # Create filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in assignment["title"])
        filename = f"{safe_title}_submissions.pdf"
        
        # Return as streaming response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
@router.get("/assignments")
async def get_student_assignments(auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Get all assignments for the current student"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Find assignments where this student is allowed
    cursor = assignments_collection.find(
        {"allowed_students": user_email},
        {"_id": 0}
    )
    
    student_assignments = []
    async for assignment in cursor:
        # Check if student has accepted this assignment
        student_record = await student_assignments_collection.find_one({
            "assignment_id": assignment["assignment_id"],
            "student_email": user_email
        })
        
    
        questions_answered = 0
        submitted = False
        submitted_at = None
        post_quiz_completed = False

        submissions_enabled = assignment.get("submissions_enabled", True)
        submission_exceptions = [e.lower() for e in assignment.get("submission_exceptions", [])]
        can_submit = submissions_enabled or (user_email in submission_exceptions)
        
        if student_record:
            questions_answered = sum(
                1 for q in student_record.get("questions", [])
                if q.get("student_solution") is not None
            )
            submitted = student_record.get("submitted", False)
            submitted_at = student_record.get("submitted_at")
            post_quiz_completed = student_record.get("post_quiz_completed", False)
        
        student_assignments.append({
            "assignment_id": assignment["assignment_id"],
            "title": assignment["title"],
            "description": assignment["description"],
            "total_questions": len(assignment["questions"]),
            "accepted": student_record is not None,
            "accepted_at": student_record["accepted_at"].isoformat() if student_record else None,
            "has_pre_quiz": assignment.get("pre_quiz_id") is not None,
            "has_post_quiz": assignment.get("post_quiz_id") is not None,
            "post_quiz_completed": post_quiz_completed,
            "questions_answered": questions_answered,
            "submitted": submitted,  
            "submitted_at": submitted_at,
            "submissions_enabled": can_submit
        })
    
    return {"assignments": student_assignments}


@router.post("/assignments/{assignment_id}/accept")
async def accept_assignment(assignment_id: str, auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Accept an assignment - requires pre-quiz completion if exists"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    user_id = user["auth0_id"]
    
    # Check if assignment exists and student is allowed
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    if user_email not in assignment.get("allowed_students", []):
        raise HTTPException(status_code=403, detail="You are not allowed to access this assignment")
    
    #  Check if pre-quiz must be completed first
    if assignment.get("pre_quiz_id"):
        pre_quiz_response = await student_quiz_responses_collection.find_one({
            "assignment_id": assignment_id,
            "student_email": user_email,
            "quiz_type": "pre"
        })
        
        if not pre_quiz_response:
            raise HTTPException(
                status_code=400,
                detail="You must complete the pre-quiz before accepting this assignment"
            )
    
    # Check if already accepted
    existing = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Assignment already accepted")
    
    # Create a conversation for each question
    questions_with_chats = []
    for idx, q in enumerate(assignment["questions"]):
        chat_id = str(uuid.uuid4())
        
        question_number = q.get('number', str(idx + 1))
        question_text = q.get('prompt_md', '')
        question_marks = q.get('marks', 0)
        
        initial_messages = create_assignment_system_prompt(
            question_number=question_number,
            question_text=question_text,
            hints=q.get("hints", [])
        )
        
        conversation_doc = {
            "chat_id": chat_id,
            "user_id": user_id,
            "messages": initial_messages,
            "summary": f"{assignment['title']} - Q{question_number}",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "is_deleted": False,
            "assignment_id": assignment_id,
            "question_id": q["question_id"],
            "is_assignment_chat": True
        }
        
        await conversations_collection.insert_one(conversation_doc)
        
        questions_with_chats.append({
            "question_id": q["question_id"],
            "number": question_number,
            "prompt_md": question_text,
            "marks": question_marks,
            "hints": q.get("hints", []),
            "chat_id": chat_id,
            "old_chats": [],
            "student_solution": None,
            "submitted_chat_id": None,
            "submitted_message_index": None,
            "submitted_at": None,
            "is_correct": None,
            "attempts": 0
        })
    
    # Create student assignment record
    student_assignment = {
        "assignment_id": assignment_id,
        "student_email": user_email,
        "accepted_at": datetime.now(timezone.utc),
        "questions": questions_with_chats,
        "post_quiz_completed": False  #  
    }
    
    await student_assignments_collection.insert_one(student_assignment)
    
    return {
        "message": "Assignment accepted successfully",
        "conversations_created": len(questions_with_chats)
    }


@router.get("/assignments/{assignment_id}")
async def get_assignment_details(assignment_id: str, auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Get assignment details with student's progress"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Get student's assignment record
    student_assignment = await student_assignments_collection.find_one(
        {
            "assignment_id": assignment_id,
            "student_email": user_email
        },
        {"_id": 0}
    )
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not accepted")
    
    # Get assignment details
    assignment = await assignments_collection.find_one(
        {"assignment_id": assignment_id},
        {"_id": 0, "allowed_students": 0}
    )
    
    # Simplified - no snapshot comparison
    questions_answered = sum(
        1 for q in student_assignment.get("questions", [])
        if q.get("student_solution") is not None
    )
    
    # ✅ ADD submission permission check
    submissions_enabled = assignment.get("submissions_enabled", True)
    submission_exceptions = [e.lower() for e in assignment.get("submission_exceptions", [])]
    can_submit = submissions_enabled or (user_email in submission_exceptions)
    
    return {
        "assignment_id": assignment_id,
        "title": assignment["title"],
        "description": assignment["description"],
        "questions": student_assignment["questions"],
        "accepted_at": student_assignment["accepted_at"].isoformat(),
        "has_pre_quiz": assignment.get("pre_quiz_id") is not None,
        "has_post_quiz": assignment.get("post_quiz_id") is not None,
        "post_quiz_completed": student_assignment.get("post_quiz_completed", False),
        "questions_answered": questions_answered,
        "total_questions": len(student_assignment["questions"]),
        "submitted": student_assignment.get("submitted", False),
        "submitted_at": student_assignment.get("submitted_at"),
        "submissions_enabled": can_submit
    }


@router.get("/assignments/{assignment_id}/questions/{question_id}/chat")
async def get_question_chat(
    assignment_id: str,
    question_id: str,
    reset: bool = False,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Get or create chat for a specific question. If reset=True, archive old chat and create new one."""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    user_id = user["auth0_id"]
    
    # Get student assignment
    student_assignment = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not accepted")
    
    # Find the question
    question_index = -1
    target_question = None
    
    for idx, question in enumerate(student_assignment["questions"]):
        if question["question_id"] == question_id:
            question_index = idx
            target_question = question
            break
    
    if target_question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Handle reset: move old chat to old_chats array
    if reset and target_question.get("chat_id"):
        old_chat_id = target_question["chat_id"]
        
        # Initialize old_chats array if not exists
        if "old_chats" not in target_question:
            target_question["old_chats"] = []
        
        # Add old chat to old_chats array
        target_question["old_chats"].append(old_chat_id)
        
        # Clear current chat_id to force creation of new one
        target_question["chat_id"] = None
        
        # Reset submission data
        target_question["attempts"] = 0
        target_question["student_solution"] = None
        target_question["submitted_chat_id"] = None
        target_question["submitted_message_index"] = None
        target_question["submitted_at"] = None
        target_question["is_correct"] = None
        
        # Update in database
        await student_assignments_collection.update_one(
            {
                "assignment_id": assignment_id,
                "student_email": user_email
            },
            {
                "$set": {
                    f"questions.{question_index}": target_question
                }
            }
        )
    
    # If chat already exists (and not resetting), return it
    if target_question.get("chat_id"):
        existing_chat = await conversations_collection.find_one({
            "chat_id": target_question["chat_id"]
        })
        
        if existing_chat:
            return {
                "chat_id": target_question["chat_id"],
                "created": False,
                "reset": False
            }
    
    # Create new chat (for reset or if missing)
    new_chat_id = str(uuid.uuid4())
    
    question_number = target_question.get('number', str(question_index + 1))
    question_text = target_question.get('prompt_md', '')
    
    #  Use helper function
    initial_messages = create_assignment_system_prompt(
        question_number=question_number,
        question_text=question_text,
        hints=target_question.get("hints", [])
    )
    
    # Get assignment for summary
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    assignment_title = assignment["title"] if assignment else "Assignment"
    
    conversation_doc = {
        "chat_id": new_chat_id,
        "user_id": user_id,
        "messages": initial_messages,
        "summary": f"{assignment_title} - Q{question_number}",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "is_deleted": False,
        "assignment_id": assignment_id,
        "question_id": question_id,
        "is_assignment_chat": True
    }
    
    await conversations_collection.insert_one(conversation_doc)
    
    # Update student assignment with new chat_id
    target_question["chat_id"] = new_chat_id
    student_assignment["questions"][question_index] = target_question
    
    await student_assignments_collection.update_one(
        {
            "assignment_id": assignment_id,
            "student_email": user_email
        },
        {
            "$set": {
                "questions": student_assignment["questions"]
            }
        }
    )
    
    return {
        "chat_id": new_chat_id,
        "created": True,
        "reset": reset
    }

@router.post("/assignments/{assignment_id}/questions/{question_id}/submit-answer")
async def submit_answer(
    assignment_id: str,
    question_id: str,
    request: SubmitAnswerRequest,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Submit a message as the final answer for a question"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Get student assignment
    student_assignment = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not accepted")
    
    await check_submission_enabled(assignment_id, user_email)

    # Find the question
    question_index = -1
    target_question = None
    
    for idx, question in enumerate(student_assignment["questions"]):
        if question["question_id"] == question_id:
            question_index = idx
            target_question = question
            break
    
    if target_question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Verify the chat belongs to this question (current or old)
    valid_chat_ids = [target_question.get("chat_id")] + target_question.get("old_chats", [])
    valid_chat_ids = [cid for cid in valid_chat_ids if cid]
    
    if request.chat_id not in valid_chat_ids:
        raise HTTPException(status_code=400, detail="Chat does not belong to this question")
    
    # ✅ FIX: Get the LATEST conversation data
    chat = await conversations_collection.find_one({
        "chat_id": request.chat_id
    })
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = chat.get("messages", [])
    
    selected_message = None
    actual_index = -1
    
    # Try exact content match first
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user" and msg.get("content") == request.message_content:
            selected_message = msg
            actual_index = idx
            break
    
    # ✅ If exact match fails, try normalized match (strip whitespace)
    if selected_message is None:
        normalized_content = request.message_content.strip()
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user" and msg.get("content", "").strip() == normalized_content:
                selected_message = msg
                actual_index = idx
                break
    
    # ✅ Still no match? Return more helpful error
    if selected_message is None:
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        raise HTTPException(
            status_code=400, 
            detail=f"Message not found in conversation. Found {len(user_messages)} user messages. Please refresh the page and try again."
        )
    
    # Update the question with the submitted answer
    target_question["student_solution"] = request.message_content
    target_question["submitted_chat_id"] = request.chat_id
    target_question["submitted_message_index"] = actual_index
    target_question["submitted_at"] = datetime.now(timezone.utc).isoformat()
    target_question["attempts"] = target_question.get("attempts", 0) + 1
    
    # Update in database
    await student_assignments_collection.update_one(
        {
            "assignment_id": assignment_id,
            "student_email": user_email
        },
        {
            "$set": {
                f"questions.{question_index}": target_question
            }
        }
    )
    
    return {
        "message": "Answer submitted successfully",
        "submitted_at": target_question["submitted_at"],
        "chat_id": request.chat_id,
        "message_index": actual_index,
        "attempts": target_question["attempts"]
    }

@router.post("/assignments/{assignment_id}/submit")
async def submit_assignment(
    assignment_id: str,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Submit assignment - just marks it as submitted (requires post-quiz completion)"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Get student assignment
    student_assignment = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not accepted")
    
    await check_submission_enabled(assignment_id, user_email)
    # Get assignment to check for post-quiz
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    # Check if post-quiz is required and completed
    has_post_quiz = assignment.get("post_quiz_id") is not None
    
    if has_post_quiz:
        post_quiz_completed = student_assignment.get("post_quiz_completed", False)
        
        if not post_quiz_completed:
            raise HTTPException(
                status_code=400,
                detail="Please complete the post-quiz before submitting your assignment"
            )
    
    # Check if already submitted
    if student_assignment.get("submitted", False):
        raise HTTPException(status_code=400, detail="Assignment already submitted")
    
    # Count questions answered
    questions_answered = sum(
        1 for q in student_assignment["questions"] 
        if q.get("student_solution") is not None
    )
    total_questions = len(student_assignment["questions"])
    
    # Mark as submitted
    await student_assignments_collection.update_one(
        {
            "assignment_id": assignment_id,
            "student_email": user_email
        },
        {
            "$set": {
                "submitted": True,
                "submitted_at": datetime.now(timezone.utc).isoformat()
            }
        }
    )
    
    return {
        "message": "Assignment submitted successfully",
        "questions_answered": questions_answered,
        "total_questions": total_questions,
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }

@router.get("/assignments/{assignment_id}/chats")
async def get_assignment_chats(assignment_id: str, auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Get all chat sessions for this assignment"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Get student's assignment record
    student_assignment = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not found or not accepted")
    
    # Return questions with their chat histories
    chats = []
    for question in student_assignment.get("questions", []):
        chats.append({
            "question_id": question["question_id"],
            "number": question.get("number", ""),
            "prompt_md": question.get("prompt_md", ""),
            "chat_id": question.get("chat_id")
        })
    
    return {"chats": chats}

@router.put("/assignment-templates/{template_id}")
async def update_assignment_template(
    template_id: str,
    request: UpdateTemplateRequest,
    user: dict = Depends(require_admin)
):
    """Update an existing assignment template (admin only)"""
    # Check if template exists
    existing = await templates_collection.find_one({"template_id": template_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Update template
    updated_doc = {
        "title": request.title,
        "description": request.description,
        "questions": [
            {
                "question_id": q.question_id or str(uuid.uuid4()),
                "number": q.number,
                "prompt_md": q.prompt_md,
                "marks": float(q.marks),
                "hints": q.hints or []
            }
            for q in request.questions
        ],
        "updated_by": user["email"],
        "updated_at": datetime.now(timezone.utc)
    }
    
    await templates_collection.update_one(
        {"template_id": template_id},
        {"$set": updated_doc}
    )
    
    return {
        "message": "Template updated successfully",
        "template_id": template_id
    }

@router.delete("/assignment-templates/{template_id}")
async def delete_assignment_template(
    template_id: str,
    user: dict = Depends(require_admin)
):
    """Delete an assignment template (admin only)"""
    # Check if template is used by any assignments
    assignment_count = await assignments_collection.count_documents({"template_id": template_id})
    if assignment_count > 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete template. It is used by {assignment_count} assignment(s). Delete those assignments first."
        )
    
    result = await templates_collection.delete_one({"template_id": template_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"message": "Template deleted successfully"}

@router.delete("/assignments/{assignment_id}")
async def delete_assignment(
    assignment_id: str,
    user: dict = Depends(require_admin)
):
    """Delete an assignment and all student progress (admin only)"""
    # Check if assignment exists
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    # Delete student assignments
    student_result = await student_assignments_collection.delete_many({"assignment_id": assignment_id})
    
    # Delete assignment conversations (soft delete)
    await conversations_collection.update_many(
        {"assignment_id": assignment_id},
        {"$set": {"is_deleted": True}}
    )
    
    # Delete assignment
    await assignments_collection.delete_one({"assignment_id": assignment_id})
    
    return {
        "message": "Assignment deleted successfully",
        "student_assignments_deleted": student_result.deleted_count
    }

# ========== QUIZ ROUTES (ADMIN) ==========

@router.post("/quiz-templates")
async def create_quiz_template(
    request: CreateQuizTemplateRequest,
    user: dict = Depends(require_admin)
):
    """Create a new quiz template (admin only)"""
    quiz_id = str(uuid.uuid4())
    
    quiz_doc = {
        "quiz_id": quiz_id,
        "title": request.title,
        "description": request.description or "",
        "questions": [
            {
                "question_id": q.question_id or str(uuid.uuid4()),
                "question_text": q.question_text,
                "options": [
                    {
                        "option_id": opt.option_id or str(uuid.uuid4()),
                        "text": opt.text,
                        "is_correct": opt.is_correct
                    }
                    for opt in q.options
                ],
                "explanation": q.explanation
            }
            for q in request.questions
        ],
        "created_by": user["email"],
        "created_at": datetime.now(timezone.utc)
    }
    
    await quiz_templates_collection.insert_one(quiz_doc)
    
    return {
        "message": "Quiz template created successfully",
        "quiz_id": quiz_id
    }

@router.get("/quiz-templates")
async def get_quiz_templates(user: dict = Depends(require_admin)):
    """Get all quiz templates (admin only)"""
    cursor = quiz_templates_collection.find({}, {"_id": 0})
    quizzes = []
    async for quiz in cursor:
        quizzes.append(quiz)
    return {"quizzes": quizzes}

@router.put("/quiz-templates/{quiz_id}")
async def update_quiz_template(
    quiz_id: str,
    request: UpdateQuizTemplateRequest,
    user: dict = Depends(require_admin)
):
    """Update a quiz template (admin only)"""
    existing = await quiz_templates_collection.find_one({"quiz_id": quiz_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Quiz template not found")
    
    updated_doc = {
        "title": request.title,
        "description": request.description or "",
        "questions": [
            {
                "question_id": q.question_id or str(uuid.uuid4()),
                "question_text": q.question_text,
                "options": [
                    {
                        "option_id": opt.option_id or str(uuid.uuid4()),
                        "text": opt.text,
                        "is_correct": opt.is_correct
                    }
                    for opt in q.options
                ],
                "explanation": q.explanation
            }
            for q in request.questions
        ],
        "updated_by": user["email"],
        "updated_at": datetime.now(timezone.utc)
    }
    
    await quiz_templates_collection.update_one(
        {"quiz_id": quiz_id},
        {"$set": updated_doc}
    )
    
    return {"message": "Quiz template updated successfully", "quiz_id": quiz_id}

@router.delete("/quiz-templates/{quiz_id}")
async def delete_quiz_template(
    quiz_id: str,
    user: dict = Depends(require_admin)
):
    """Delete a quiz template (admin only)"""
    # Check if quiz is used in any assignments
    pre_quiz_count = await assignments_collection.count_documents({"pre_quiz_id": quiz_id})
    post_quiz_count = await assignments_collection.count_documents({"post_quiz_id": quiz_id})
    
    if pre_quiz_count > 0 or post_quiz_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete quiz. Used in {pre_quiz_count + post_quiz_count} assignment(s)."
        )
    
    result = await quiz_templates_collection.delete_one({"quiz_id": quiz_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Quiz template not found")
    
    return {"message": "Quiz template deleted successfully"}

# ========== QUIZ ROUTES (STUDENT) ==========

@router.get("/assignments/{assignment_id}/pre-quiz")
async def get_pre_quiz(
    assignment_id: str,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Get pre-quiz for an assignment"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Get assignment
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    if user_email not in assignment.get("allowed_students", []):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if not assignment.get("pre_quiz_id"):
        raise HTTPException(status_code=404, detail="No pre-quiz for this assignment")
    
    # Get quiz template
    quiz = await quiz_templates_collection.find_one(
        {"quiz_id": assignment["pre_quiz_id"]},
        {"_id": 0}
    )
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Pre-quiz template not found")
    
    # Hide correct answers
    quiz_data = quiz.copy()
    for q in quiz_data["questions"]:
        for opt in q["options"]:
            opt.pop("is_correct", None)
        q.pop("explanation", None)
    
    # Check if already completed
    existing_response = await student_quiz_responses_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email,
        "quiz_type": "pre"
    })
    
    return {
        "quiz": quiz_data,
        "completed": existing_response is not None,
        "score": existing_response.get("score") if existing_response else None
    }

@router.post("/assignments/{assignment_id}/pre-quiz/submit")
async def submit_pre_quiz(
    assignment_id: str,
    answers: List[SubmitQuizAnswerRequest],
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Submit pre-quiz answers"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    # Get assignment
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    if not assignment.get("pre_quiz_id"):
        raise HTTPException(status_code=404, detail="No pre-quiz for this assignment")
    
    # Get quiz template
    quiz = await quiz_templates_collection.find_one({"quiz_id": assignment["pre_quiz_id"]})
    if not quiz:
        raise HTTPException(status_code=404, detail="Pre-quiz template not found")
    
    # Check if already completed
    existing = await student_quiz_responses_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email,
        "quiz_type": "pre"
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Pre-quiz already completed")
    
    # Grade the quiz
    correct_count = 0
    total_questions = len(quiz["questions"])
    detailed_results = []
    
    answer_map = {ans.question_id: ans.selected_option_id for ans in answers}
    
    for q in quiz["questions"]:
        selected_option_id = answer_map.get(q["question_id"])
        correct_option = next((opt for opt in q["options"] if opt["is_correct"]), None)
        
        is_correct = selected_option_id == correct_option["option_id"] if correct_option else False
        if is_correct:
            correct_count += 1
        
        detailed_results.append({
            "question_id": q["question_id"],
            "selected_option_id": selected_option_id,
            "is_correct": is_correct,
            "correct_option_id": correct_option["option_id"] if correct_option else None,
            "explanation": q.get("explanation")
        })
    
    score = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    # Save response
    response_doc = {
        "assignment_id": assignment_id,
        "student_email": user_email,
        "quiz_type": "pre",
        "quiz_id": assignment["pre_quiz_id"],
        "answers": [ans.dict() for ans in answers],
        "score": score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "submitted_at": datetime.now(timezone.utc)
    }
    
    await student_quiz_responses_collection.insert_one(response_doc)
    
    return {
        "message": "Pre-quiz submitted successfully",
        "score": score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "results": detailed_results
    }


@router.get("/assignments/{assignment_id}/post-quiz")
async def get_post_quiz(
    assignment_id: str,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Get post-quiz for an assignment"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    #  REMOVED: Check if student has completed all questions
    # Just check if student has accepted the assignment
    student_assignment = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not accepted")
    
    #  REMOVED: all_completed check
    
    # Get assignment
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment or not assignment.get("post_quiz_id"):
        raise HTTPException(status_code=404, detail="No post-quiz for this assignment")
    
    # Get quiz template
    quiz = await quiz_templates_collection.find_one(
        {"quiz_id": assignment["post_quiz_id"]},
        {"_id": 0}
    )
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Post-quiz template not found")
    
    # Hide correct answers
    quiz_data = quiz.copy()
    for q in quiz_data["questions"]:
        for opt in q["options"]:
            opt.pop("is_correct", None)
        q.pop("explanation", None)
    
    # Check if already completed
    existing_response = await student_quiz_responses_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email,
        "quiz_type": "post"
    })
    
    return {
        "quiz": quiz_data,
        "completed": existing_response is not None,
        "score": existing_response.get("score") if existing_response else None
    }

@router.post("/assignments/{assignment_id}/post-quiz/submit")
async def submit_post_quiz(
    assignment_id: str,
    answers: List[SubmitQuizAnswerRequest],
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Submit post-quiz answers"""
    user = await get_current_user(auth)
    user_email = user["email"].lower()
    
    #  REMOVED: Same validation as get_post_quiz
    # Just check if student has accepted the assignment
    student_assignment = await student_assignments_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email
    })
    
    if not student_assignment:
        raise HTTPException(status_code=404, detail="Assignment not accepted")
    
    #  REMOVED: all_completed check
    
    # Get assignment
    assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
    if not assignment or not assignment.get("post_quiz_id"):
        raise HTTPException(status_code=404, detail="No post-quiz for this assignment")
    
    # Get quiz template
    quiz = await quiz_templates_collection.find_one({"quiz_id": assignment["post_quiz_id"]})
    if not quiz:
        raise HTTPException(status_code=404, detail="Post-quiz template not found")
    
    # Check if already completed
    existing = await student_quiz_responses_collection.find_one({
        "assignment_id": assignment_id,
        "student_email": user_email,
        "quiz_type": "post"
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Post-quiz already completed")
    
    # Grade the quiz (same logic as pre-quiz)
    correct_count = 0
    total_questions = len(quiz["questions"])
    detailed_results = []
    
    answer_map = {ans.question_id: ans.selected_option_id for ans in answers}
    
    for q in quiz["questions"]:
        selected_option_id = answer_map.get(q["question_id"])
        correct_option = next((opt for opt in q["options"] if opt["is_correct"]), None)
        
        is_correct = selected_option_id == correct_option["option_id"] if correct_option else False
        if is_correct:
            correct_count += 1
        
        detailed_results.append({
            "question_id": q["question_id"],
            "selected_option_id": selected_option_id,
            "is_correct": is_correct,
            "correct_option_id": correct_option["option_id"] if correct_option else None,
            "explanation": q.get("explanation")
        })
    
    score = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    # Save response
    response_doc = {
        "assignment_id": assignment_id,
        "student_email": user_email,
        "quiz_type": "post",
        "quiz_id": assignment["post_quiz_id"],
        "answers": [ans.dict() for ans in answers],
        "score": score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "submitted_at": datetime.now(timezone.utc)
    }
    
    await student_quiz_responses_collection.insert_one(response_doc)
    
    # Mark assignment as fully completed
    await student_assignments_collection.update_one(
        {
            "assignment_id": assignment_id,
            "student_email": user_email
        },
        {
            "$set": {
                "completed_at": datetime.now(timezone.utc),
                "post_quiz_completed": True
            }
        }
    )
    
    return {
        "message": "Post-quiz submitted successfully! Assignment completed.",
        "score": score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "results": detailed_results
    }

@router.post("/assignments/{assignment_id}/export-grades-csv")
async def export_grades_csv(
    assignment_id: str,
    user: dict = Depends(require_admin)
):
    """Export assignment grades as CSV with latest submission timestamp in US-East timezone (admin only)"""
    try:
        from io import StringIO
        import csv
        from datetime import datetime
        from zoneinfo import ZoneInfo
        
        # Get assignment details
        assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        # Get all student submissions
        cursor = student_assignments_collection.find(
            {"assignment_id": assignment_id},
            {"_id": 0}
        )
        
        # Create CSV in memory
        output = StringIO()
        csv_writer = csv.writer(output)
        
        # Write header - only 3 columns
        header = ["Student Name", "Email", "Latest Submission (US-East)"]
        csv_writer.writerow(header)
        
        # Process each student
        async for student_assignment in cursor:
            # Get student details
            user_doc = await users_collection.find_one({"email": student_assignment["student_email"]})
            student_name = user_doc.get("username", "Unknown") if user_doc else "Unknown"
            student_email = student_assignment["student_email"]
            
            # Find the latest submission time across all questions
            latest_timestamp = None
            latest_dt = None
            
            for question in student_assignment["questions"]:
                if question.get("submitted_at"):
                    try:
                        # Parse ISO format datetime
                        dt_utc = datetime.fromisoformat(question["submitted_at"].replace('Z', '+00:00'))
                        
                        # Track the latest timestamp
                        if latest_dt is None or dt_utc > latest_dt:
                            latest_dt = dt_utc
                    except Exception as e:
                        print(f"Error parsing timestamp: {e}")
                        continue
            
            # Convert latest timestamp to US-East timezone
            if latest_dt:
                try:
                    dt_eastern = latest_dt.astimezone(ZoneInfo("America/New_York"))
                    latest_timestamp = dt_eastern.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception as e:
                    latest_timestamp = "Error converting timezone"
            else:
                latest_timestamp = "Not submitted"
            
            csv_writer.writerow([
                student_name,
                student_email,
                latest_timestamp
            ])
        
        # Get CSV content
        csv_content = output.getvalue()
        output.close()
        
        if not csv_content or csv_content.count('\n') <= 1:
            raise HTTPException(status_code=404, detail="No submissions found for this assignment")
        
        # Create filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in assignment["title"])
        filename = f"{safe_title}_grades.csv"
        
        # Return as streaming response
        from io import BytesIO
        csv_bytes = BytesIO(csv_content.encode('utf-8'))
        
        return StreamingResponse(
            csv_bytes,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating CSV: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")


# ========== GRADING ROUTES ==========

@router.get("/assignments/{assignment_id}/submissions")
async def get_assignment_submissions(
    assignment_id: str,
    user: dict = Depends(require_grader)
):
    """Get all student submissions for an assignment (grader/admin only)"""
    try:
        # Get assignment details
        assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        # Get all student submissions
        cursor = student_assignments_collection.find(
            {"assignment_id": assignment_id},
            {"_id": 0}
        )
        
        submissions = []
        async for student_assignment in cursor:
            # Get student details
            user_doc = await users_collection.find_one({"email": student_assignment["student_email"]})
            student_name = user_doc.get("username", "Unknown") if user_doc else "Unknown"
            
            # Calculate latest submission time
            latest_submitted_at = None
            for question in student_assignment.get("questions", []):
                if question.get("submitted_at"):
                    if latest_submitted_at is None or question["submitted_at"] > latest_submitted_at:
                        latest_submitted_at = question["submitted_at"]
            
            submissions.append({
                "student_email": student_assignment["student_email"],
                "student_name": student_name,
                "accepted": student_assignment.get("accepted", False),
                "submitted": student_assignment.get("submitted", False),
                "submitted_at": student_assignment.get("submitted_at"),
                "latest_answer_time": latest_submitted_at,
                "questions_answered": student_assignment.get("questions_answered", 0),
                "total_questions": student_assignment.get("total_questions", 0),
                "pre_quiz_completed": student_assignment.get("pre_quiz_completed", False),
                "post_quiz_completed": student_assignment.get("post_quiz_completed", False)
            })
        
        return {
            "assignment": {
                "assignment_id": assignment["assignment_id"],
                "title": assignment["title"],
                "due_date": assignment.get("due_date"),
                "total_questions": len(assignment.get("questions", []))
            },
            "submissions": submissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting submissions: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get submissions: {str(e)}")


@router.get("/assignments/{assignment_id}/submissions/{student_email}")
async def get_student_submission(
    assignment_id: str,
    student_email: str,
    user: dict = Depends(require_grader)
):
    """Get a specific student's submission details (grader/admin only)"""
    try:
        # Get assignment details
        assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        # Get student submission
        student_assignment = await student_assignments_collection.find_one(
            {
                "assignment_id": assignment_id,
                "student_email": student_email
            },
            {"_id": 0}
        )
        
        if not student_assignment:
            raise HTTPException(status_code=404, detail="Student submission not found")
        
        # Get student details
        user_doc = await users_collection.find_one({"email": student_email})
        student_name = user_doc.get("username", "Unknown") if user_doc else "Unknown"
        
        # Enrich questions with chat history
        enriched_questions = []
        for question in student_assignment.get("questions", []):
            question_data = question.copy()
            
            # Get chat history if chat_id exists
            if question.get("chat_id"):
                chat = await conversations_collection.find_one(
                    {"conversation_id": question["chat_id"]},
                    {"_id": 0}
                )
                if chat:
                    question_data["chat_history"] = chat.get("messages", [])
            
            enriched_questions.append(question_data)
        
        return {
            "assignment": {
                "assignment_id": assignment["assignment_id"],
                "title": assignment["title"],
                "instructions_md": assignment.get("instructions_md", ""),
                "due_date": assignment.get("due_date")
            },
            "student": {
                "email": student_email,
                "name": student_name
            },
            "submission": {
                "accepted": student_assignment.get("accepted", False),
                "submitted": student_assignment.get("submitted", False),
                "submitted_at": student_assignment.get("submitted_at"),
                "questions_answered": student_assignment.get("questions_answered", 0),
                "total_questions": student_assignment.get("total_questions", 0),
                "questions": enriched_questions
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting student submission: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get student submission: {str(e)}")