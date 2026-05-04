#!/usr/bin/env python3
"""
Feature Extraction Script for ML Integrity Detection

This script processes exported MongoDB data and extracts features
for training integrity detection models.

Usage:
    python extract_features.py --input ./ --output ./features

Input files expected:
    - assignments.json
    - conversations.json
    - student_assignments.json
    - telemetry.json

Output files:
    - features.csv          : Main feature matrix
    - features.parquet      : Same data, efficient format
    - feature_metadata.json : Feature descriptions and stats
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import statistics

# Try to import pandas/numpy, provide helpful error if missing
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required.")
    print("Install with: pip install pandas numpy pyarrow")
    exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns for content analysis
FORMAL_PHRASE_PATTERNS = [
    r'\bfurthermore\b',
    r'\bin conclusion\b',
    r'\bmoreover\b',
    r'\bnevertheless\b',
    r'\bconsequently\b',
    r'\btherefore\b',
    r'\bthus\b',
    r'\bhence\b',
    r'\baccordingly\b',
]

HEDGING_PATTERNS = [
    r'\bperhaps\b',
    r'\bit might be\b',
    r'\bone could argue\b',
    r'\bit is worth noting\b',
    r'\binterestingly\b',
    r'\barguably\b',
    r'\bpotentially\b',
]

AI_ARTIFACT_PATTERNS = [
    r'\bas an AI\b',
    r'\bI cannot\b',
    r"\bI don'?t have feelings\b",
    r"\bI'?m an AI\b",
    r'\blanguage model\b',
    r'\bI apologize\b',
    r'\bI\'?d be happy to\b',
]

# AI punctuation patterns (em-dashes, en-dashes, and semicolons)
AI_PUNCTUATION_PATTERNS = [
    r'—',           # Em-dash (AI loves these)
    r'–',           # En-dash
    r';',           # Semicolons (AI overuses these)
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_data(input_dir: str) -> Dict[str, Any]:
    """Load all exported JSON files."""
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    data = {}
    
    files = [
        ('assignments', 'assignments.json'),
        ('conversations', 'conversations.json'),
        ('student_assignments', 'student_assignments.json'),
        ('telemetry', 'telemetry.json'),
    ]
    
    for key, filename in files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing required file: {filepath}")
        
        data[key] = load_json(filepath)
        print(f"  ✓ {filename}: {len(data[key])} records")
    
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def count_patterns(text: str, patterns: List[str]) -> int:
    """Count occurrences of regex patterns in text."""
    if not text:
        return 0
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return a / b if b != 0 else default


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Safe mean with default for empty list."""
    return statistics.mean(values) if values else default


def safe_std(values: List[float], default: float = 0.0) -> float:
    """Safe standard deviation with default for insufficient data."""
    return statistics.stdev(values) if len(values) > 1 else default


def calculate_trend_slope(values: List[float]) -> float:
    """Calculate slope of a value series over index positions."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp string."""
    if not ts:
        return None
    try:
        # Handle various ISO formats
        ts = ts.replace('Z', '+00:00')
        if '.' in ts:
            # Truncate microseconds if too long
            parts = ts.split('.')
            if '+' in parts[1]:
                micro, tz = parts[1].split('+')
                ts = f"{parts[0]}.{micro[:6]}+{tz}"
            elif '-' in parts[1] and len(parts[1]) > 6:
                micro, tz = parts[1].rsplit('-', 1)
                ts = f"{parts[0]}.{micro[:6]}-{tz}"
        return datetime.fromisoformat(ts)
    except:
        return None


def calculate_bullet_ratio(text: str) -> float:
    """Calculate ratio of bullet-pointed lines."""
    if not text:
        return 0.0
    lines = text.strip().split('\n')
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if re.match(r'^\s*[-*•]\s|^\s*\d+[.)]\s', line))
    return bullet_lines / len(lines)


def calculate_vocabulary_richness(text: str) -> float:
    """Calculate type-token ratio (unique words / total words)."""
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def calculate_avg_word_length(text: str) -> float:
    """Calculate average word length."""
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def calculate_avg_sentence_length(text: str) -> float:
    """Calculate average sentence length in words."""
    if not text:
        return 0.0
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    
    word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return safe_mean(word_counts)


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL FEATURE EXTRACTION (from telemetry)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_behavioral_features(telemetry_events: List[Dict]) -> Dict[str, Any]:
    """Extract behavioral features from telemetry events for a single chat."""
    
    features = {}
    
    # Separate events by type
    paste_events = [e for e in telemetry_events if e.get('event_type') == 'PASTE']
    keystroke_events = [e for e in telemetry_events if e.get('event_type') == 'KEYSTROKE_BATCH']
    message_events = [e for e in telemetry_events if e.get('event_type') == 'MESSAGE_SEND']
    focus_loss_events = [e for e in telemetry_events if e.get('event_type') == 'FOCUS_LOSS']
    focus_gain_events = [e for e in telemetry_events if e.get('event_type') == 'FOCUS_GAIN']
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASTE FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    features['paste_count'] = len(paste_events)
    
    chars_pasted_list = [
        e.get('paste_data', {}).get('char_count', 0) 
        for e in paste_events
    ]
    features['total_chars_pasted'] = sum(chars_pasted_list)
    features['max_single_paste'] = max(chars_pasted_list) if chars_pasted_list else 0
    
    # Get total keystrokes for ratio calculations
    total_keystrokes = sum(
        e.get('keystroke_data', {}).get('key_count', 0) 
        for e in keystroke_events
    )
    
    # Total chars from message_send events
    total_chars_typed = sum(
        e.get('message_send_data', {}).get('total_keystrokes', 0)
        for e in message_events
    )
    
    # Use the larger of the two keystroke counts
    effective_keystrokes = max(total_keystrokes, total_chars_typed)
    
    features['paste_ratio'] = safe_div(
        features['total_chars_pasted'],
        features['total_chars_pasted'] + effective_keystrokes
    )
    features['paste_to_keystroke_ratio'] = safe_div(
        features['total_chars_pasted'], 
        effective_keystrokes
    )
    features['paste_per_message'] = safe_div(
        features['paste_count'], 
        len(message_events)
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # KEYSTROKE FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    features['total_keystrokes'] = effective_keystrokes
    
    total_backspaces = sum(
        e.get('keystroke_data', {}).get('backspace_count', 0) 
        for e in keystroke_events
    )
    # Also check message_send events for backspaces
    total_backspaces_msg = sum(
        e.get('message_send_data', {}).get('total_backspaces', 0)
        for e in message_events
    )
    features['total_backspaces'] = max(total_backspaces, total_backspaces_msg)
    
    features['backspace_ratio'] = safe_div(
        features['total_backspaces'], 
        features['total_keystrokes']
    )
    
    # Typing speed from keystroke batches
    typing_speeds = [
        e.get('keystroke_data', {}).get('typing_speed_cpm', 0)
        for e in keystroke_events
        if e.get('keystroke_data', {}).get('typing_speed_cpm', 0) > 0
    ]
    features['avg_typing_speed_cpm'] = safe_mean(typing_speeds)
    features['typing_speed_std'] = safe_std(typing_speeds)
    features['typing_speed_cv'] = safe_div(
        features['typing_speed_std'],
        features['avg_typing_speed_cpm']
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPOSITION TIME FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    composition_times = [
        e.get('message_send_data', {}).get('composition_time_ms', 0)
        for e in message_events
        if e.get('message_send_data', {}).get('composition_time_ms', 0) > 0
    ]
    
    features['total_composition_time_ms'] = sum(composition_times)
    features['avg_composition_time_ms'] = safe_mean(composition_times)
    features['min_composition_time_ms'] = min(composition_times) if composition_times else 0
    features['max_composition_time_ms'] = max(composition_times) if composition_times else 0
    
    # Chars per message from message_send
    chars_per_msg = []
    input_lengths = []
    question_mark_counts = []
    sentence_counts = []
    edit_pause_counts = []
    first_key_to_send_times = []
    focus_return_to_send_times = []
    for e in message_events:
        msg_data = e.get('message_send_data', {})
        keystrokes = msg_data.get('total_keystrokes', 0)
        pasted = msg_data.get('chars_pasted', 0)
        chars_per_msg.append(keystrokes + pasted)
        input_lengths.append(msg_data.get('message_input_length_at_send', 0) or 0)
        question_mark_counts.append(msg_data.get('message_question_mark_count', 0) or 0)
        sentence_counts.append(msg_data.get('message_sentence_count', 0) or 0)
        edit_pause_counts.append(msg_data.get('message_edit_pause_count', 0) or 0)

        first_key_to_send = msg_data.get('message_first_key_to_send_ms')
        if isinstance(first_key_to_send, (int, float)) and first_key_to_send >= 0:
            first_key_to_send_times.append(first_key_to_send)

        focus_return_to_send = msg_data.get('message_send_after_focus_return_ms')
        if isinstance(focus_return_to_send, (int, float)) and focus_return_to_send >= 0:
            focus_return_to_send_times.append(focus_return_to_send)
    
    total_chars_in_messages = sum(chars_per_msg)
    total_input_chars = sum(input_lengths)
    features['composition_per_char_ms'] = safe_div(
        features['total_composition_time_ms'],
        total_chars_in_messages
    )

    # Message-level process and light text-structure features
    features['avg_edit_pause_count_per_message'] = safe_mean(edit_pause_counts)
    features['total_edit_pause_count'] = sum(edit_pause_counts)
    features['first_key_to_send_avg_ms'] = safe_mean(first_key_to_send_times)
    features['first_key_to_send_min_ms'] = min(first_key_to_send_times) if first_key_to_send_times else 0
    features['focus_return_to_send_ms_avg'] = safe_mean(focus_return_to_send_times)
    features['focus_return_to_send_ms_min'] = min(focus_return_to_send_times) if focus_return_to_send_times else 0
    features['message_question_mark_total'] = sum(question_mark_counts)
    features['question_mark_rate_per_100_chars'] = safe_div(
        features['message_question_mark_total'] * 100,
        total_input_chars
    )
    features['message_sentence_total'] = sum(sentence_counts)
    features['sentence_density_per_message'] = safe_div(
        features['message_sentence_total'],
        len(message_events)
    )
    
    # Idle time
    idle_times = [
        e.get('keystroke_data', {}).get('idle_time_ms', 0)
        for e in keystroke_events
    ]
    features['total_idle_time_ms'] = sum(idle_times)
    
    # ─────────────────────────────────────────────────────────────────────────
    # FOCUS FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    features['focus_loss_count'] = len(focus_loss_events)
    
    # Time away from focus_gain events
    times_away = [
        e.get('focus_data', {}).get('duration_away_ms', 0) or 0
        for e in focus_gain_events
        if e.get('focus_data', {}).get('duration_away_ms')
    ]
    features['total_time_away_ms'] = sum(times_away)
    features['avg_time_away_ms'] = safe_mean(times_away)
    
    # Focus losses from message_send data
    focus_losses_in_msgs = sum(
        e.get('message_send_data', {}).get('focus_losses', 0)
        for e in message_events
    )
    features['focus_loss_before_submit'] = focus_losses_in_msgs
    
    # ─────────────────────────────────────────────────────────────────────────
    # SESSION FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    # Calculate session duration from timestamps
    timestamps = []
    for e in telemetry_events:
        ts = parse_timestamp(e.get('client_timestamp') or e.get('server_timestamp'))
        if ts:
            timestamps.append(ts)
    
    if len(timestamps) >= 2:
        timestamps.sort()
        duration = (timestamps[-1] - timestamps[0]).total_seconds() * 1000
        features['session_duration_ms'] = duration
    else:
        features['session_duration_ms'] = 0
    
    features['events_per_minute'] = safe_div(
        len(telemetry_events) * 60000,
        features['session_duration_ms']
    )
    features['message_count'] = len(message_events)
    features['telemetry_event_count'] = len(telemetry_events)
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT FEATURE EXTRACTION (from messages)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_content_features(messages: List[Dict]) -> Dict[str, Any]:
    """Extract content features from conversation messages."""
    
    features = {}
    
    # Filter by role
    student_msgs = [m for m in messages if m.get('role') == 'user']
    ai_msgs = [m for m in messages if m.get('role') == 'assistant']

    inter_turn_from_meta = [
        m.get('inter_turn_response_ms')
        for m in student_msgs
        if isinstance(m.get('inter_turn_response_ms'), (int, float)) and m.get('inter_turn_response_ms') >= 0
    ]
    lexical_richness_from_meta = [
        m.get('lexical_richness')
        for m in student_msgs
        if isinstance(m.get('lexical_richness'), (int, float))
    ]
    
    # Get all student text
    student_texts = [m.get('content', '') for m in student_msgs]
    all_student_text = ' '.join(student_texts)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONVERSATION STRUCTURE
    # ─────────────────────────────────────────────────────────────────────────
    features['turn_count'] = len(student_msgs)
    features['total_messages'] = len(messages)
    features['student_to_ai_ratio'] = safe_div(len(student_msgs), len(ai_msgs))
    features['conversation_depth'] = min(len(student_msgs), len(ai_msgs))
    features['has_submitted'] = any(
        'final answer' in m.get('content', '').lower() 
        for m in student_msgs
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # MESSAGE LENGTH PATTERNS
    # ─────────────────────────────────────────────────────────────────────────
    student_lengths = [len(m.get('content', '')) for m in student_msgs]
    
    features['first_message_length'] = student_lengths[0] if student_lengths else 0
    features['avg_message_length'] = safe_mean(student_lengths)
    features['max_message_length'] = max(student_lengths) if student_lengths else 0
    features['message_length_std'] = safe_std(student_lengths)
    features['total_student_chars'] = sum(student_lengths)
    
    # Message length growth (last vs first)
    if len(student_lengths) >= 2 and student_lengths[0] > 0:
        features['message_length_growth'] = student_lengths[-1] / student_lengths[0]
    else:
        features['message_length_growth'] = 1.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE TIME PATTERNS
    # ─────────────────────────────────────────────────────────────────────────
    response_times = []
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user' and i > 0:
            prev_msg = messages[i - 1]
            if prev_msg.get('role') == 'assistant':
                curr_ts = parse_timestamp(msg.get('timestamp'))
                prev_ts = parse_timestamp(prev_msg.get('timestamp'))
                if curr_ts and prev_ts:
                    diff_ms = (curr_ts - prev_ts).total_seconds() * 1000
                    if diff_ms > 0:
                        response_times.append(diff_ms)
    
    features['avg_response_time_ms'] = safe_mean(response_times)
    features['min_response_time_ms'] = min(response_times) if response_times else 0
    features['response_time_variance'] = safe_std(response_times)
    features['user_response_time_trend_slope'] = calculate_trend_slope(response_times)
    features['avg_inter_turn_response_ms_from_meta'] = safe_mean(inter_turn_from_meta)
    
    # Fast long responses (suspicious pattern)
    fast_long_count = 0
    for i, msg in enumerate(student_msgs):
        msg_len = len(msg.get('content', ''))
        if msg_len > 300 and i < len(response_times):
            if response_times[i] < 60000:  # Less than 60 seconds
                fast_long_count += 1
    features['fast_long_response_count'] = fast_long_count
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENGAGEMENT SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    features['question_count'] = all_student_text.count('?')
    
    # ─────────────────────────────────────────────────────────────────────────
    # LLM ARTIFACT DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    features['bullet_point_ratio'] = calculate_bullet_ratio(all_student_text)
    features['formal_phrase_count'] = count_patterns(all_student_text, FORMAL_PHRASE_PATTERNS)
    features['hedging_phrase_count'] = count_patterns(all_student_text, HEDGING_PATTERNS)
    features['ai_artifact_count'] = count_patterns(all_student_text, AI_ARTIFACT_PATTERNS)
    features['ai_punctuation_count'] = count_patterns(all_student_text, AI_PUNCTUATION_PATTERNS)
    
    # Em-dash ratio (em-dashes per 1000 chars)
    emdash_count = len(re.findall(r'—', all_student_text))
    features['emdash_ratio'] = safe_div(emdash_count * 1000, len(all_student_text))
    
    # Semicolon ratio (semicolons per 1000 chars)
    semicolon_count = len(re.findall(r';', all_student_text))
    features['semicolon_ratio'] = safe_div(semicolon_count * 1000, len(all_student_text))
    
    # LaTeX usage
    latex_count = all_student_text.count('$')
    features['latex_formula_count'] = latex_count
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOCABULARY FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    features['vocabulary_richness'] = calculate_vocabulary_richness(all_student_text)
    features['avg_message_lexical_richness'] = safe_mean(lexical_richness_from_meta)
    features['avg_word_length'] = calculate_avg_word_length(all_student_text)
    features['avg_sentence_length'] = calculate_avg_sentence_length(all_student_text)
    
    # Combined formality score
    formality_indicators = (
        features['formal_phrase_count'] +
        features['hedging_phrase_count'] +
        features['bullet_point_ratio'] * 10
    )
    features['formal_language_score'] = formality_indicators
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(data: Dict[str, Any]) -> pd.DataFrame:
    """Build complete feature matrix from loaded data."""
    
    print(f"\n{'='*60}")
    print("EXTRACTING FEATURES")
    print(f"{'='*60}")
    
    # Index data by chat_id for fast lookup
    conversations_by_chat = {c['chat_id']: c for c in data['conversations']}
    
    # Group telemetry by chat_id
    telemetry_by_chat = defaultdict(list)
    for event in data['telemetry']:
        chat_id = event.get('chat_id')
        if chat_id:
            telemetry_by_chat[chat_id].append(event)
    
    # Build question lookup from assignments
    questions_by_id = {}
    for assignment in data['assignments']:
        for q in assignment.get('questions', []):
            questions_by_id[q['question_id']] = {
                'question_text': q.get('prompt_md', ''),
                'question_number': q.get('number', ''),
                'marks': q.get('marks', 0),
            }
    
    # Process each student-question submission
    rows = []
    
    for student_assignment in data['student_assignments']:
        student_email = student_assignment.get('student_email', '')
        assignment_id = student_assignment.get('assignment_id', '')
        
        for question in student_assignment.get('questions', []):
            chat_id = question.get('chat_id')
            if not chat_id:
                continue
            
            question_id = question.get('question_id', '')
            
            # Get conversation
            conversation = conversations_by_chat.get(chat_id, {})
            messages = conversation.get('messages', [])
            
            # Get telemetry
            telemetry_events = telemetry_by_chat.get(chat_id, [])
            
            # Get question info
            q_info = questions_by_id.get(question_id, {})
            
            print(f"  Processing: {student_email} - Q{question.get('number', '?')} ({len(telemetry_events)} telemetry events)")
            
            # ─────────────────────────────────────────────────────────────────
            # Build feature row
            # ─────────────────────────────────────────────────────────────────
            row = {}
            
            # Identifiers
            row['chat_id'] = chat_id
            row['user_id'] = conversation.get('user_id', '')
            row['student_email'] = student_email
            row['assignment_id'] = assignment_id
            row['question_id'] = question_id
            row['question_number'] = question.get('number', '')
            
            # Extract behavioral features
            behavioral = extract_behavioral_features(telemetry_events)
            row.update(behavioral)
            
            # Extract content features
            content = extract_content_features(messages)
            row.update(content)
            
            # Metadata (handle None values)
            question_text = q_info.get('question_text') or ''
            row['question_text'] = question_text[:500]  # Truncate
            
            submitted_solution = question.get('student_solution') or ''
            row['submitted_solution'] = submitted_solution[:1000]
            
            row['submitted_at'] = question.get('submitted_at') or ''
            row['submitted_message_index'] = question.get('submitted_message_index')
            row['is_correct'] = question.get('is_correct')
            row['attempts'] = question.get('attempts', 0)
            
            # Placeholder for label (to be filled manually)
            row['label'] = ''
            row['label_type'] = ''
            
            rows.append(row)
    
    df = pd.DataFrame(rows)

    if not df.empty:
        # Relative lexical shift against each student-assignment baseline.
        # Uses simple normalized distance for robust behavior on small samples.
        baseline_vr = df.groupby(['student_email', 'assignment_id'])['vocabulary_richness'].transform('mean')
        baseline_awl = df.groupby(['student_email', 'assignment_id'])['avg_word_length'].transform('mean')

        vr_delta = (df['vocabulary_richness'] - baseline_vr).abs()
        awl_delta = (df['avg_word_length'] - baseline_awl).abs()
        df['lexical_shift_score'] = vr_delta + (awl_delta / baseline_awl.clip(lower=1.0))
    else:
        df['lexical_shift_score'] = []

    print(f"\n  ✓ Extracted features for {len(df)} submissions")
    
    return df


def generate_feature_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate metadata about extracted features."""
    
    # Define feature groups
    behavioral_features = [
        'paste_count', 'total_chars_pasted', 'paste_ratio', 'max_single_paste',
        'paste_to_keystroke_ratio', 'paste_per_message',
        'total_keystrokes', 'total_backspaces', 'backspace_ratio',
        'avg_typing_speed_cpm', 'typing_speed_std', 'typing_speed_cv',
        'total_composition_time_ms', 'avg_composition_time_ms',
        'min_composition_time_ms', 'max_composition_time_ms',
        'composition_per_char_ms', 'total_idle_time_ms',
        'avg_edit_pause_count_per_message', 'total_edit_pause_count',
        'first_key_to_send_avg_ms', 'first_key_to_send_min_ms',
        'focus_return_to_send_ms_avg', 'focus_return_to_send_ms_min',
        'message_question_mark_total', 'question_mark_rate_per_100_chars',
        'message_sentence_total', 'sentence_density_per_message',
        'focus_loss_count', 'total_time_away_ms', 'avg_time_away_ms',
        'focus_loss_before_submit', 'session_duration_ms', 'events_per_minute',
        'message_count', 'telemetry_event_count',
    ]
    
    content_features = [
        'turn_count', 'total_messages', 'student_to_ai_ratio',
        'conversation_depth', 'has_submitted',
        'first_message_length', 'avg_message_length', 'max_message_length',
        'message_length_std', 'total_student_chars', 'message_length_growth',
        'avg_response_time_ms', 'min_response_time_ms', 'response_time_variance',
        'user_response_time_trend_slope', 'avg_inter_turn_response_ms_from_meta',
        'fast_long_response_count',
        'question_count',
        'bullet_point_ratio', 'formal_phrase_count', 'hedging_phrase_count',
        'ai_artifact_count', 'ai_punctuation_count',
        'emdash_ratio', 'semicolon_ratio', 'latex_formula_count',
        'vocabulary_richness', 'avg_message_lexical_richness', 'avg_word_length', 'avg_sentence_length',
        'formal_language_score', 'lexical_shift_score',
    ]
    
    metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'total_samples': len(df),
        'feature_counts': {
            'behavioral': len(behavioral_features),
            'content': len(content_features),
            'total': len(behavioral_features) + len(content_features),
        },
        'feature_groups': {
            'behavioral': behavioral_features,
            'content': content_features,
        },
        'feature_statistics': {},
    }
    
    # Calculate statistics for numeric features
    all_features = behavioral_features + content_features
    for feat in all_features:
        if feat in df.columns:
            col = df[feat]
            if pd.api.types.is_numeric_dtype(col):
                metadata['feature_statistics'][feat] = {
                    'mean': float(col.mean()) if not col.isna().all() else None,
                    'std': float(col.std()) if not col.isna().all() else None,
                    'min': float(col.min()) if not col.isna().all() else None,
                    'max': float(col.max()) if not col.isna().all() else None,
                    'missing': int(col.isna().sum()),
                }
    
    return metadata


def save_features(df: pd.DataFrame, metadata: Dict, output_dir: str):
    """Save feature matrix and metadata."""
    
    print(f"\n{'='*60}")
    print("SAVING FEATURES")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'features.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✓ features.csv ({len(df)} rows)")
    
    # Save Parquet
    try:
        parquet_path = os.path.join(output_dir, 'features.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"  ✓ features.parquet")
    except Exception as e:
        print(f"  ⚠ Could not save parquet: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'feature_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ feature_metadata.json")
    
    print(f"\n  All files saved to: {os.path.abspath(output_dir)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from exported ML data"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='.',
        help='Input directory containing exported JSON files (default: current dir)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./features',
        help='Output directory for feature files (default: ./features)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print(f"Input directory:  {os.path.abspath(args.input)}")
    print(f"Output directory: {os.path.abspath(args.output)}")
    
    # Load data
    data = load_all_data(args.input)
    
    # Build feature matrix
    df = build_feature_matrix(data)
    
    # Generate metadata
    metadata = generate_feature_metadata(df)
    
    # Save
    save_features(df, metadata, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total submissions: {len(df)}")
    print(f"  Behavioral features: {metadata['feature_counts']['behavioral']}")
    print(f"  Content features: {metadata['feature_counts']['content']}")
    print(f"  Total features: {metadata['feature_counts']['total']}")
    
    # Show sample of key features
    print(f"\n  Sample feature values (first row):")
    if len(df) > 0:
        key_features = ['paste_ratio', 'backspace_ratio', 'turn_count', 'first_message_length']
        for feat in key_features:
            if feat in df.columns:
                print(f"    {feat}: {df[feat].iloc[0]:.3f}" if isinstance(df[feat].iloc[0], float) else f"    {feat}: {df[feat].iloc[0]}")
    
    print(f"\n✓ Feature extraction complete!")
    print(f"\nNext steps:")
    print(f"  1. Open features/features.csv")
    print(f"  2. Add 'label' column: 'genuine' or 'suspicious'")
    print(f"  3. Add 'label_type' column: 'manual' or 'heuristic'")


if __name__ == "__main__":
    main()
