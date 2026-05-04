"""
Telemetry Routes for ML Behavioral Analysis

This module handles the collection of client-side interaction telemetry
for post-submission analysis of authentic engagement patterns.

Events tracked:
- PASTE: Text paste events with character count and content hash
- KEYSTROKE_BATCH: Aggregated keystroke metrics (typing speed, backspaces, idle time)
- FOCUS_LOSS: User switched away from the ALAASKA tab
- FOCUS_GAIN: User returned to the ALAASKA tab
- MESSAGE_SEND: Message submission event with composition metrics
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from backend.auth import get_current_user, http_bearer
from backend.db_mongo import telemetry_collection
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone
import hashlib
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== PYDANTIC MODELS ==========

class PasteEventData(BaseModel):
    """Data for paste events"""
    char_count: int = Field(..., ge=0, description="Number of characters pasted")
    word_count: int = Field(..., ge=0, description="Number of words pasted")
    content_hash: str = Field(..., description="SHA256 hash of pasted content for matching")


class KeystrokeBatchData(BaseModel):
    """Aggregated keystroke data sent periodically"""
    key_count: int = Field(default=0, ge=0, description="Total keys pressed in batch")
    backspace_count: int = Field(default=0, ge=0, description="Number of backspaces (revision indicator)")
    typing_speed_cpm: float = Field(default=0, ge=0, description="Characters per minute")
    idle_time_ms: int = Field(default=0, ge=0, description="Time focused but not typing (ms)")
    batch_duration_ms: int = Field(default=0, ge=0, description="Duration of this batch (ms)")


class FocusEventData(BaseModel):
    """Data for focus/blur events"""
    duration_away_ms: Optional[int] = Field(default=None, ge=0, description="Time spent away from tab (ms)")


class MessageSendData(BaseModel):
    """Data captured when user sends a message"""
    composition_time_ms: int = Field(..., ge=0, description="Total time composing message (ms)")
    total_keystrokes: int = Field(default=0, ge=0, description="Total keystrokes for this message")
    total_backspaces: int = Field(default=0, ge=0, description="Total backspaces for this message")
    paste_count: int = Field(default=0, ge=0, description="Number of paste events during composition")
    chars_pasted: int = Field(default=0, ge=0, description="Total characters pasted")
    focus_losses: int = Field(default=0, ge=0, description="Times user left tab during composition")
    message_edit_pause_count: int = Field(default=0, ge=0, description="Number of edit pauses above threshold")
    message_first_key_to_send_ms: Optional[int] = Field(default=None, ge=0, description="Time from first keypress to send (ms)")
    message_input_length_at_send: int = Field(default=0, ge=0, description="Input length at send time (chars)")
    message_send_after_focus_return_ms: Optional[int] = Field(default=None, ge=0, description="Time from latest focus gain to send (ms)")
    message_question_mark_count: int = Field(default=0, ge=0, description="Number of question marks in submitted input")
    message_sentence_count: int = Field(default=0, ge=0, description="Estimated sentence count in submitted input")


class TelemetryEvent(BaseModel):
    """Single telemetry event from the frontend"""
    event_type: str = Field(..., description="PASTE | KEYSTROKE_BATCH | FOCUS_LOSS | FOCUS_GAIN | MESSAGE_SEND")
    timestamp: str = Field(..., description="ISO timestamp when event occurred on client")
    chat_id: str = Field(..., description="Chat session ID")
    assignment_id: Optional[str] = Field(default=None, description="Assignment ID if assignment chat")
    question_id: Optional[str] = Field(default=None, description="Question ID if assignment chat")
    session_id: str = Field(..., description="Browser session ID for grouping events")
    
    # Event-specific data (only one will be populated based on event_type)
    paste_data: Optional[PasteEventData] = None
    keystroke_data: Optional[KeystrokeBatchData] = None
    focus_data: Optional[FocusEventData] = None
    message_send_data: Optional[MessageSendData] = None


class TelemetryBatch(BaseModel):
    """Batch of telemetry events for efficient transmission"""
    events: List[TelemetryEvent] = Field(..., min_length=1, max_length=100)


# ========== HELPER FUNCTIONS ==========

def hash_content(content: str) -> str:
    """Create SHA256 hash of content for matching without storing full text"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# ========== ENDPOINTS ==========

@router.post("/telemetry/event")
async def record_telemetry_event(
    event: TelemetryEvent,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """
    Record a single telemetry event from the frontend.
    
    Used for immediate high-priority events like paste detection.
    """
    user = await get_current_user(auth)
    user_id = user["auth0_id"]
    
    # Validate event type
    valid_types = ["PASTE", "KEYSTROKE_BATCH", "FOCUS_LOSS", "FOCUS_GAIN", "MESSAGE_SEND"]
    if event.event_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid event_type. Must be one of: {valid_types}")
    
    # Build telemetry document
    telemetry_doc = {
        "user_id": user_id,
        "event_type": event.event_type,
        "client_timestamp": event.timestamp,
        "server_timestamp": datetime.now(timezone.utc).isoformat(),
        "chat_id": event.chat_id,
        "assignment_id": event.assignment_id,
        "question_id": event.question_id,
        "session_id": event.session_id,
    }
    
    # Add event-specific data
    if event.event_type == "PASTE" and event.paste_data:
        telemetry_doc["paste_data"] = event.paste_data.model_dump()
    elif event.event_type == "KEYSTROKE_BATCH" and event.keystroke_data:
        telemetry_doc["keystroke_data"] = event.keystroke_data.model_dump()
    elif event.event_type in ["FOCUS_LOSS", "FOCUS_GAIN"] and event.focus_data:
        telemetry_doc["focus_data"] = event.focus_data.model_dump()
    elif event.event_type == "MESSAGE_SEND" and event.message_send_data:
        telemetry_doc["message_send_data"] = event.message_send_data.model_dump()
    
    try:
        await telemetry_collection.insert_one(telemetry_doc)
        return {"status": "recorded", "event_type": event.event_type}
    except Exception as e:
        logger.error(f"Failed to record telemetry event: {e}")
        raise HTTPException(status_code=500, detail="Failed to record telemetry")


@router.post("/telemetry/batch")
async def record_telemetry_batch(
    batch: TelemetryBatch,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """
    Record a batch of telemetry events.
    
    Used for periodic transmission of accumulated events (e.g., keystroke batches).
    More efficient than individual calls for high-frequency events.
    """
    user = await get_current_user(auth)
    user_id = user["auth0_id"]
    
    valid_types = ["PASTE", "KEYSTROKE_BATCH", "FOCUS_LOSS", "FOCUS_GAIN", "MESSAGE_SEND"]
    server_timestamp = datetime.now(timezone.utc).isoformat()
    
    documents = []
    for event in batch.events:
        if event.event_type not in valid_types:
            continue  # Skip invalid events in batch
        
        telemetry_doc = {
            "user_id": user_id,
            "event_type": event.event_type,
            "client_timestamp": event.timestamp,
            "server_timestamp": server_timestamp,
            "chat_id": event.chat_id,
            "assignment_id": event.assignment_id,
            "question_id": event.question_id,
            "session_id": event.session_id,
        }
        
        # Add event-specific data
        if event.event_type == "PASTE" and event.paste_data:
            telemetry_doc["paste_data"] = event.paste_data.model_dump()
        elif event.event_type == "KEYSTROKE_BATCH" and event.keystroke_data:
            telemetry_doc["keystroke_data"] = event.keystroke_data.model_dump()
        elif event.event_type in ["FOCUS_LOSS", "FOCUS_GAIN"] and event.focus_data:
            telemetry_doc["focus_data"] = event.focus_data.model_dump()
        elif event.event_type == "MESSAGE_SEND" and event.message_send_data:
            telemetry_doc["message_send_data"] = event.message_send_data.model_dump()
        
        documents.append(telemetry_doc)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No valid events in batch")
    
    try:
        result = await telemetry_collection.insert_many(documents)
        return {
            "status": "recorded",
            "events_recorded": len(result.inserted_ids)
        }
    except Exception as e:
        logger.error(f"Failed to record telemetry batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to record telemetry batch")


@router.get("/telemetry/chat/{chat_id}")
async def get_chat_telemetry(
    chat_id: str,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """
    Get all telemetry events for a specific chat session.
    
    Used for post-submission analysis and instructor review.
    Requires grader/admin access for other users' data.
    """
    user = await get_current_user(auth)
    user_id = user["auth0_id"]
    is_grader = user.get("is_grader", False)
    
    # Build query - graders can see all, students only their own
    if is_grader:
        query = {"chat_id": chat_id}
    else:
        query = {"chat_id": chat_id, "user_id": user_id}
    
    cursor = telemetry_collection.find(query, {"_id": 0}).sort("client_timestamp", 1)
    
    events = []
    async for doc in cursor:
        events.append(doc)
    
    # Calculate summary metrics
    summary = {
        "total_events": len(events),
        "paste_events": sum(1 for e in events if e["event_type"] == "PASTE"),
        "total_chars_pasted": sum(
            e.get("paste_data", {}).get("char_count", 0) 
            for e in events if e["event_type"] == "PASTE"
        ),
        "focus_losses": sum(1 for e in events if e["event_type"] == "FOCUS_LOSS"),
        "total_time_away_ms": sum(
            e.get("focus_data", {}).get("duration_away_ms", 0) or 0
            for e in events if e["event_type"] == "FOCUS_GAIN"
        ),
        "keystroke_batches": sum(1 for e in events if e["event_type"] == "KEYSTROKE_BATCH"),
        "total_backspaces": sum(
            e.get("keystroke_data", {}).get("backspace_count", 0)
            for e in events if e["event_type"] == "KEYSTROKE_BATCH"
        )
    }
    
    return {
        "chat_id": chat_id,
        "events": events,
        "summary": summary
    }
