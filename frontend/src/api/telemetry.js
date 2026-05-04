/**
 * Telemetry Service for ML Behavioral Analysis
 * 
 * This module tracks user interactions for post-submission analysis
 * of authentic engagement patterns in ALAASKA conversations.
 * 
 * Events Tracked:
 * - PASTE: Text paste events
 * - KEYSTROKE_BATCH: Aggregated typing metrics
 * - FOCUS_LOSS/GAIN: Tab switching behavior
 * - MESSAGE_SEND: Per-message composition metrics
 */

import api from './axios';

const PAUSE_THRESHOLD_MS = 2000;

// Simple hash function for content fingerprinting (no external dependency)
const simpleHash = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
};

// Generate a unique session ID for this browser session
const SESSION_ID = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

// Telemetry state per chat session
const telemetryState = {
  chatId: null,
  assignmentId: null,
  questionId: null,
  
  // Keystroke tracking
  keyCount: 0,
  backspaceCount: 0,
  typingStartTime: null,
  lastKeystrokeTime: null,
  idleTime: 0,
  
  // Paste tracking
  pasteCount: 0,
  totalCharsPasted: 0,
  
  // Focus tracking
  focusLostTime: null,
  focusLossCount: 0,
  
  // Message composition tracking
  messageStartTime: null,
  messageFirstKeypressTime: null,
  messageKeystrokes: 0,
  messageBackspaces: 0,
  messagePasteCount: 0,
  messageCharsPasted: 0,
  messageFocusLosses: 0,
  messageEditPauseCount: 0,
  lastFocusGainTime: null,
  
  // Batch queue
  eventQueue: [],
  batchInterval: null,
};

/**
 * Initialize telemetry for a chat session
 */
export const initTelemetry = (chatId, assignmentId = null, questionId = null) => {
  telemetryState.chatId = chatId;
  telemetryState.assignmentId = assignmentId;
  telemetryState.questionId = questionId;
  
  // Reset counters
  resetMessageCounters();
  
  // Start batch sending interval (every 10 seconds)
  if (telemetryState.batchInterval) {
    clearInterval(telemetryState.batchInterval);
  }
  telemetryState.batchInterval = setInterval(flushEventQueue, 10000);
  
  console.log('[Telemetry] Initialized for chat:', chatId);
};

/**
 * Clean up telemetry (call on unmount)
 */
export const cleanupTelemetry = () => {
  if (telemetryState.batchInterval) {
    clearInterval(telemetryState.batchInterval);
    telemetryState.batchInterval = null;
  }
  flushEventQueue(); // Send any remaining events
};

/**
 * Reset counters for a new message composition
 */
const resetMessageCounters = () => {
  telemetryState.messageStartTime = Date.now();
  telemetryState.messageFirstKeypressTime = null;
  telemetryState.messageKeystrokes = 0;
  telemetryState.messageBackspaces = 0;
  telemetryState.messagePasteCount = 0;
  telemetryState.messageCharsPasted = 0;
  telemetryState.messageFocusLosses = 0;
  telemetryState.messageEditPauseCount = 0;
  telemetryState.lastFocusGainTime = null;
  telemetryState.keyCount = 0;
  telemetryState.backspaceCount = 0;
  telemetryState.typingStartTime = null;
  telemetryState.idleTime = 0;
};

/**
 * Create a base telemetry event object
 */
const createBaseEvent = (eventType) => ({
  event_type: eventType,
  timestamp: new Date().toISOString(),
  chat_id: telemetryState.chatId,
  assignment_id: telemetryState.assignmentId,
  question_id: telemetryState.questionId,
  session_id: SESSION_ID,
});

/**
 * Send a single high-priority event immediately
 */
const sendEvent = async (event) => {
  if (!telemetryState.chatId) return;
  
  try {
    await api.post('/telemetry/event', event);
  } catch (err) {
    console.warn('[Telemetry] Failed to send event:', err.message);
    // Queue it for retry in next batch
    telemetryState.eventQueue.push(event);
  }
};

/**
 * Add event to batch queue
 */
const queueEvent = (event) => {
  telemetryState.eventQueue.push(event);
};

/**
 * Flush event queue to server
 */
const flushEventQueue = async () => {
  if (telemetryState.eventQueue.length === 0) return;
  if (!telemetryState.chatId) return;
  
  const events = [...telemetryState.eventQueue];
  telemetryState.eventQueue = [];
  
  try {
    await api.post('/telemetry/batch', { events });
  } catch (err) {
    console.warn('[Telemetry] Failed to send batch:', err.message);
    // Put events back in queue for retry
    telemetryState.eventQueue.push(...events);
  }
};

// ==================== EVENT HANDLERS ====================

/**
 * Handle paste event - HIGH PRIORITY, send immediately
 */
export const handlePaste = (e) => {
  if (!telemetryState.chatId) return;
  
  const pastedText = e.clipboardData?.getData('text') || '';
  const charCount = pastedText.length;
  const wordCount = pastedText.trim().split(/\s+/).filter(w => w).length;
  
  // Update counters
  telemetryState.pasteCount++;
  telemetryState.totalCharsPasted += charCount;
  telemetryState.messagePasteCount++;
  telemetryState.messageCharsPasted += charCount;
  
  // Create and send event immediately (paste is high-priority indicator)
  const event = {
    ...createBaseEvent('PASTE'),
    paste_data: {
      char_count: charCount,
      word_count: wordCount,
      content_hash: simpleHash(pastedText), // Hash for matching without storing full text
    },
  };
  
  sendEvent(event);
  console.log('[Telemetry] Paste detected:', charCount, 'chars');
};

/**
 * Handle keydown event - batched
 */
export const handleKeyDown = (e) => {
  if (!telemetryState.chatId) return;
  
  const now = Date.now();
  
  // Track idle time (time since last keystroke)
  if (telemetryState.lastKeystrokeTime) {
    const timeSinceLastKey = now - telemetryState.lastKeystrokeTime;
    if (timeSinceLastKey > PAUSE_THRESHOLD_MS) { // More than threshold = idle/edit pause
      telemetryState.idleTime += timeSinceLastKey;
      telemetryState.messageEditPauseCount++;
    }
  }
  
  // Start typing timer if not started
  if (!telemetryState.typingStartTime) {
    telemetryState.typingStartTime = now;
  }
  if (!telemetryState.messageFirstKeypressTime) {
    telemetryState.messageFirstKeypressTime = now;
  }
  
  telemetryState.lastKeystrokeTime = now;
  telemetryState.keyCount++;
  telemetryState.messageKeystrokes++;
  
  // Track backspaces specifically (important for authenticity detection)
  if (e.key === 'Backspace' || e.key === 'Delete') {
    telemetryState.backspaceCount++;
    telemetryState.messageBackspaces++;
  }
};

/**
 * Send keystroke batch (called periodically)
 */
export const sendKeystrokeBatch = () => {
  if (!telemetryState.chatId) return;
  if (telemetryState.keyCount === 0) return;
  
  const now = Date.now();
  const duration = telemetryState.typingStartTime 
    ? now - telemetryState.typingStartTime 
    : 0;
  
  // Calculate typing speed (characters per minute)
  const typingSpeedCpm = duration > 0 
    ? (telemetryState.keyCount / duration) * 60000 
    : 0;
  
  const event = {
    ...createBaseEvent('KEYSTROKE_BATCH'),
    keystroke_data: {
      key_count: telemetryState.keyCount,
      backspace_count: telemetryState.backspaceCount,
      typing_speed_cpm: Math.round(typingSpeedCpm),
      idle_time_ms: Math.round(telemetryState.idleTime),
      batch_duration_ms: Math.round(duration),
    },
  };
  
  queueEvent(event);
  
  // Reset batch counters (but keep message-level counters)
  telemetryState.keyCount = 0;
  telemetryState.backspaceCount = 0;
  telemetryState.typingStartTime = null;
  telemetryState.idleTime = 0;
};

/**
 * Handle focus loss (user switched tabs)
 */
export const handleFocusLoss = () => {
  if (!telemetryState.chatId) return;
  
  telemetryState.focusLostTime = Date.now();
  telemetryState.focusLossCount++;
  telemetryState.messageFocusLosses++;
  
  const event = createBaseEvent('FOCUS_LOSS');
  queueEvent(event);
  
  console.log('[Telemetry] Focus lost');
};

/**
 * Handle focus gain (user returned to tab)
 */
export const handleFocusGain = () => {
  if (!telemetryState.chatId) return;

  telemetryState.lastFocusGainTime = Date.now();
  
  let durationAway = null;
  if (telemetryState.focusLostTime) {
    durationAway = Date.now() - telemetryState.focusLostTime;
    telemetryState.focusLostTime = null;
  }
  
  const event = {
    ...createBaseEvent('FOCUS_GAIN'),
    focus_data: {
      duration_away_ms: durationAway,
    },
  };
  
  // If user was away for a significant time, send immediately
  if (durationAway && durationAway > 5000) {
    sendEvent(event);
  } else {
    queueEvent(event);
  }
  
  console.log('[Telemetry] Focus gained, was away:', durationAway, 'ms');
};

/**
 * Handle message send - captures full composition metrics
 */
export const handleMessageSend = (inputText = '') => {
  if (!telemetryState.chatId) return;
  
  // First, flush any remaining keystroke data
  sendKeystrokeBatch();
  
  const now = Date.now();
  const compositionTime = telemetryState.messageStartTime
    ? now - telemetryState.messageStartTime
    : 0;
  const firstKeyToSendMs = telemetryState.messageFirstKeypressTime
    ? now - telemetryState.messageFirstKeypressTime
    : null;
  const focusReturnToSendMs = telemetryState.lastFocusGainTime
    ? now - telemetryState.lastFocusGainTime
    : null;
  const inputLength = inputText.length;
  const questionMarkCount = (inputText.match(/\?/g) || []).length;
  const sentenceCount = inputText
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(Boolean)
    .length;
  
  const event = {
    ...createBaseEvent('MESSAGE_SEND'),
    message_send_data: {
      composition_time_ms: compositionTime,
      total_keystrokes: telemetryState.messageKeystrokes,
      total_backspaces: telemetryState.messageBackspaces,
      paste_count: telemetryState.messagePasteCount,
      chars_pasted: telemetryState.messageCharsPasted,
      focus_losses: telemetryState.messageFocusLosses,
      message_edit_pause_count: telemetryState.messageEditPauseCount,
      message_first_key_to_send_ms: firstKeyToSendMs,
      message_input_length_at_send: inputLength,
      message_send_after_focus_return_ms: focusReturnToSendMs,
      message_question_mark_count: questionMarkCount,
      message_sentence_count: sentenceCount,
    },
  };
  
  // Send immediately since message is being submitted
  sendEvent(event);
  
  // Reset for next message
  resetMessageCounters();
  
  console.log('[Telemetry] Message sent, composition time:', compositionTime, 'ms');
};

/**
 * Get current session ID (for debugging)
 */
export const getSessionId = () => SESSION_ID;

// Named export for default
const telemetryService = {
  initTelemetry,
  cleanupTelemetry,
  handlePaste,
  handleKeyDown,
  sendKeystrokeBatch,
  handleFocusLoss,
  handleFocusGain,
  handleMessageSend,
  getSessionId,
};

export default telemetryService;
