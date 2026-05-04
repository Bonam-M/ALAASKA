#!/usr/bin/env python3
"""
Data Export Script for ML Integrity Detection System

This script exports data from MongoDB to JSON files for local ML model training.
Run this on the server, then SCP the output files to your local machine.

Usage:
    # Export ALL data (all assignments)
    python export_data_for_ml.py --all --output ./ml_export
    
    # Export data for a specific assignment by title (partial match)
    python export_data_for_ml.py --assignment "Assignment-2" --output ./ml_export
    
    # Export data for a specific assignment by ID
    python export_data_for_ml.py --assignment-id "abc123-def456" --output ./ml_export
    
    # List all assignments (to find the right one)
    python export_data_for_ml.py --list-assignments

Output Files:
    - conversations.json      : All chat conversations with messages
    - telemetry.json          : Behavioral telemetry events
    - student_assignments.json: Student submission data
    - assignments.json        : Assignment definitions with questions
    - export_metadata.json    : Export info (timestamp, filters, counts)
"""

import asyncio
import argparse
import json
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId, json_util
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_CLIENT = os.getenv("MONGODB_CLIENT")  # Database name

if not MONGODB_URL:
    raise ValueError("Missing MONGODB_URL in .env")
if not MONGODB_CLIENT:
    raise ValueError("Missing MONGODB_CLIENT in .env")


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles MongoDB types."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json_util.default(obj)


def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    
    result = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_doc(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_doc(item) if isinstance(item, dict) else
                str(item) if isinstance(item, ObjectId) else
                item.isoformat() if isinstance(item, datetime) else
                item
                for item in value
            ]
        else:
            result[key] = value
    return result


async def get_db_connection():
    """Create MongoDB connection."""
    client = AsyncIOMotorClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=60000
    )
    db = client[MONGODB_CLIENT]
    
    # Test connection
    await client.admin.command('ping')
    print(f"✓ Connected to MongoDB database: {MONGODB_CLIENT}")
    
    return client, db


async def list_assignments(db) -> List[Dict]:
    """List all assignments with basic info."""
    assignments_collection = db["assignments"]
    
    cursor = assignments_collection.find(
        {},
        {"assignment_id": 1, "title": 1, "created_at": 1, "allowed_students": 1}
    ).sort("created_at", -1)
    
    assignments = []
    async for doc in cursor:
        assignments.append({
            "assignment_id": doc.get("assignment_id"),
            "title": doc.get("title"),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
            "student_count": len(doc.get("allowed_students", []))
        })
    
    return assignments


async def export_all_data(db, output_dir: str) -> Dict[str, int]:
    """Export all data from all collections."""
    print("\n" + "="*60)
    print("EXPORTING ALL DATA")
    print("="*60)
    
    counts = {}
    
    # 1. Export assignments (definitions)
    print("\n[1/4] Exporting assignments...")
    assignments_collection = db["assignments"]
    assignments = []
    async for doc in assignments_collection.find({}):
        assignments.append(serialize_doc(doc))
    counts["assignments"] = len(assignments)
    print(f"  → Found {len(assignments)} assignments")
    
    # Get all assignment IDs for filtering
    assignment_ids = [a.get("assignment_id") for a in assignments]
    
    # 2. Export student_assignments
    print("\n[2/4] Exporting student assignments...")
    student_assignments_collection = db["student_assignments"]
    student_assignments = []
    async for doc in student_assignments_collection.find({}):
        student_assignments.append(serialize_doc(doc))
    counts["student_assignments"] = len(student_assignments)
    print(f"  → Found {len(student_assignments)} student assignment records")
    
    # 3. Export conversations (assignment chats only)
    print("\n[3/4] Exporting conversations...")
    conversations_collection = db["conversations"]
    conversations = []
    async for doc in conversations_collection.find({"is_assignment_chat": True}):
        conversations.append(serialize_doc(doc))
    counts["conversations"] = len(conversations)
    print(f"  → Found {len(conversations)} assignment conversations")
    
    # Get all chat IDs for telemetry filtering
    chat_ids = [c.get("chat_id") for c in conversations]
    
    # 4. Export telemetry
    print("\n[4/4] Exporting telemetry...")
    telemetry_collection = db["interaction_telemetry"]
    telemetry = []
    
    # Export in batches to handle large collections
    batch_size = 10000
    total_telemetry = 0
    
    if chat_ids:
        cursor = telemetry_collection.find({"chat_id": {"$in": chat_ids}})
        async for doc in cursor:
            telemetry.append(serialize_doc(doc))
            total_telemetry += 1
            if total_telemetry % batch_size == 0:
                print(f"  → Processed {total_telemetry} telemetry events...")
    
    counts["telemetry"] = len(telemetry)
    print(f"  → Found {len(telemetry)} telemetry events")
    
    # Save all files
    save_exports(output_dir, assignments, student_assignments, conversations, telemetry, 
                 filter_info={"type": "all"})
    
    return counts


async def export_assignment_data(
    db, 
    output_dir: str, 
    assignment_title: Optional[str] = None,
    assignment_id: Optional[str] = None
) -> Dict[str, int]:
    """Export data for a specific assignment."""
    print("\n" + "="*60)
    print(f"EXPORTING ASSIGNMENT DATA")
    print("="*60)
    
    assignments_collection = db["assignments"]
    
    # Find the assignment
    if assignment_id:
        assignment = await assignments_collection.find_one({"assignment_id": assignment_id})
        if not assignment:
            raise ValueError(f"Assignment not found with ID: {assignment_id}")
    elif assignment_title:
        # Case-insensitive partial match
        assignment = await assignments_collection.find_one({
            "title": {"$regex": assignment_title, "$options": "i"}
        })
        if not assignment:
            raise ValueError(f"Assignment not found matching title: {assignment_title}")
    else:
        raise ValueError("Must provide either assignment_title or assignment_id")
    
    assignment = serialize_doc(assignment)
    target_assignment_id = assignment["assignment_id"]
    
    print(f"\n→ Found assignment: {assignment.get('title')}")
    print(f"  ID: {target_assignment_id}")
    print(f"  Students: {len(assignment.get('allowed_students', []))}")
    
    counts = {}
    
    # 1. Assignment definition
    print("\n[1/4] Exporting assignment definition...")
    assignments = [assignment]
    counts["assignments"] = 1
    
    # 2. Export student_assignments for this assignment
    print("\n[2/4] Exporting student assignments...")
    student_assignments_collection = db["student_assignments"]
    student_assignments = []
    async for doc in student_assignments_collection.find({"assignment_id": target_assignment_id}):
        student_assignments.append(serialize_doc(doc))
    counts["student_assignments"] = len(student_assignments)
    print(f"  → Found {len(student_assignments)} student assignment records")
    
    # 3. Export conversations for this assignment
    print("\n[3/4] Exporting conversations...")
    conversations_collection = db["conversations"]
    conversations = []
    async for doc in conversations_collection.find({
        "is_assignment_chat": True,
        "assignment_id": target_assignment_id
    }):
        conversations.append(serialize_doc(doc))
    counts["conversations"] = len(conversations)
    print(f"  → Found {len(conversations)} conversations")
    
    # Get all chat IDs
    chat_ids = [c.get("chat_id") for c in conversations]
    
    # Also get old_chats from questions in student_assignments
    for sa in student_assignments:
        for q in sa.get("questions", []):
            if q.get("chat_id"):
                chat_ids.append(q["chat_id"])
            chat_ids.extend(q.get("old_chats", []))
    
    chat_ids = list(set(chat_ids))  # Deduplicate
    
    # 4. Export telemetry for these chats
    print("\n[4/4] Exporting telemetry...")
    telemetry_collection = db["interaction_telemetry"]
    telemetry = []
    
    if chat_ids:
        async for doc in telemetry_collection.find({"chat_id": {"$in": chat_ids}}):
            telemetry.append(serialize_doc(doc))
    
    counts["telemetry"] = len(telemetry)
    print(f"  → Found {len(telemetry)} telemetry events")
    
    # Save all files
    save_exports(output_dir, assignments, student_assignments, conversations, telemetry,
                 filter_info={
                     "type": "single_assignment",
                     "assignment_id": target_assignment_id,
                     "assignment_title": assignment.get("title")
                 })
    
    return counts


def save_exports(
    output_dir: str,
    assignments: List[Dict],
    student_assignments: List[Dict],
    conversations: List[Dict],
    telemetry: List[Dict],
    filter_info: Dict
):
    """Save all exported data to JSON files."""
    print("\n" + "="*60)
    print("SAVING FILES")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each collection
    files = [
        ("assignments.json", assignments),
        ("student_assignments.json", student_assignments),
        ("conversations.json", conversations),
        ("telemetry.json", telemetry),
    ]
    
    for filename, data in files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=JSONEncoder)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f" {filename}: {len(data)} records ({size_mb:.2f} MB)")
    
    # Save metadata
    metadata = {
        "export_schema_version": "2.0.0",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "filter": filter_info,
        "counts": {
            "assignments": len(assignments),
            "student_assignments": len(student_assignments),
            "conversations": len(conversations),
            "telemetry": len(telemetry)
        },
        "files": [f[0] for f in files]
    }
    
    metadata_path = os.path.join(output_dir, "export_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ export_metadata.json")
    
    print(f"\n✓ All files saved to: {os.path.abspath(output_dir)}")


async def main():
    parser = argparse.ArgumentParser(
        description="Export MongoDB data for ML integrity detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all assignments
  python export_data_for_ml.py --list-assignments
  
  # Export all data
  python export_data_for_ml.py --all --output ./ml_export
  
  # Export specific assignment by title (partial match)
  python export_data_for_ml.py --assignment "Assignment-2" --output ./ml_export
  
  # Export specific assignment by ID
  python export_data_for_ml.py --assignment-id "abc-123" --output ./ml_export
        """
    )
    
    parser.add_argument(
        "--list-assignments", 
        action="store_true",
        help="List all assignments and exit"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Export all data from all assignments"
    )
    parser.add_argument(
        "--assignment", 
        type=str,
        help="Export data for assignment matching this title (partial, case-insensitive)"
    )
    parser.add_argument(
        "--assignment-id", 
        type=str,
        help="Export data for assignment with this exact ID"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="./ml_export",
        help="Output directory for exported files (default: ./ml_export)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.list_assignments, args.all, args.assignment, args.assignment_id]):
        parser.print_help()
        print("\n Error: Must specify --list-assignments, --all, --assignment, or --assignment-id")
        return
    
    # Connect to database
    try:
        client, db = await get_db_connection()
    except Exception as e:
        print(f" Failed to connect to MongoDB: {e}")
        return
    
    try:
        # List assignments
        if args.list_assignments:
            print("\n" + "="*60)
            print("AVAILABLE ASSIGNMENTS")
            print("="*60)
            assignments = await list_assignments(db)
            
            if not assignments:
                print("No assignments found.")
                return
            
            print(f"\n{'ID':<40} {'Title':<30} {'Students':<10} {'Created'}")
            print("-" * 100)
            for a in assignments:
                print(f"{a['assignment_id']:<40} {a['title']:<30} {a['student_count']:<10} {a['created_at'] or 'N/A'}")
            
            print(f"\nTotal: {len(assignments)} assignments")
            return
        
        # Export all data
        if args.all:
            counts = await export_all_data(db, args.output)
        
        # Export specific assignment
        elif args.assignment or args.assignment_id:
            counts = await export_assignment_data(
                db, 
                args.output,
                assignment_title=args.assignment,
                assignment_id=args.assignment_id
            )
        
        # Print summary
        print("\n" + "="*60)
        print("EXPORT COMPLETE")
        print("="*60)
        print(f"\nExported:")
        for collection, count in counts.items():
            print(f"  • {collection}: {count} records")
        
        print(f"\nFiles saved to: {os.path.abspath(args.output)}")
        print("\nTo copy to local machine:")
        print(f"  scp -r user@server:{os.path.abspath(args.output)} ./local_ml_data/")
        
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
