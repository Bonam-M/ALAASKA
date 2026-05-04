import React, { useState, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';
import api from '../api/axios';
import LatexEditor from './LatexEditor';
import '../styles/assignment_chat.css';
import {
  initTelemetry,
  cleanupTelemetry,
  handlePaste as telemetryPaste,
  handleKeyDown as telemetryKeyDown,
  handleFocusLoss,
  handleFocusGain,
  handleMessageSend,
} from '../api/telemetry';

export default function ChatInterface({ chatId, messages, input, setInput, sendMessage, onNavigateToAssignment }) {
  const [metadata, setMetadata] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [activeMenu, setActiveMenu] = useState(null);
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0 });
  const [useLatexEditor, setUseLatexEditor] = useState(false);
  const messagesEndRef = useRef(null);
  const menuButtonRefs = useRef({});

  useEffect(() => {
    if (chatId) {
      loadChatMetadata();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  // Initialize telemetry when metadata is loaded (for assignment chats)
  useEffect(() => {
    if (chatId && metadata?.is_assignment_chat) {
      initTelemetry(chatId, metadata.assignment_id, metadata.question_id);
    } else if (chatId) {
      // Also track regular chats (but with null assignment/question)
      initTelemetry(chatId, null, null);
    }
    
    return () => {
      cleanupTelemetry();
    };
  }, [chatId, metadata?.is_assignment_chat, metadata?.assignment_id, metadata?.question_id]);

  // Set up focus/blur event listeners for tab switching detection
  useEffect(() => {
    const onFocus = () => handleFocusGain();
    const onBlur = () => handleFocusLoss();
    
    window.addEventListener('focus', onFocus);
    window.addEventListener('blur', onBlur);
    
    return () => {
      window.removeEventListener('focus', onFocus);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadChatMetadata = async () => {
    try {
      const res = await api.get(`/conversation/${chatId}`);
      const meta = res.data.metadata;
      setMetadata(meta);

      if (meta?.is_assignment_chat && meta?.assignment_id) {
        try {
          const assignmentRes = await api.get(`/assignments/${meta.assignment_id}`);
          setMetadata(prev => ({
            ...prev,
            assignment_title: assignmentRes.data.title
          }));
        } catch (err) {
          console.error('Failed to load assignment title:', err);
        }
      }
    } catch (err) {
      console.error('Failed to load chat metadata:', err);
    }
  };

  const handleMarkAsAnswer = async (messageContent) => {
    if (!metadata?.is_assignment_chat) return;
    
    if (metadata?.submissions_disabled) {
      alert('Submissions are currently disabled for this assignment.');
      return;
    }

    if (!window.confirm('Mark this message as your final answer? This will replace any previous submission.')) {
      return;
    }

    setSubmitting(true);
    setActiveMenu(null);
    
    try {
      const res = await api.post(
        `/assignments/${metadata.assignment_id}/questions/${metadata.question_id}/submit-answer`,
        {
          chat_id: chatId,
          message_index: 0,
          message_content: messageContent
        }
      );

      alert(`Answer submitted successfully!\nAttempt #${res.data.attempts}\nYou can view it in the Assignments page.`);
      loadChatMetadata();
    } catch (err) {
      console.error('Failed to submit answer:', err);
      alert(err.response?.data?.detail || 'Failed to submit answer');
    } finally {
      setSubmitting(false);
    }
  };

  const handleNavigateToAssignment = () => {
    if (metadata?.assignment_id && metadata?.question_id) {
      onNavigateToAssignment(metadata.assignment_id, metadata.question_id);
    }
  };

  const handleKeyDown = (e) => {
    // Track keystroke for telemetry (only if chatId exists)
    if (chatId) {
      telemetryKeyDown(e);
    }
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendWithTelemetry();
    }
  };

  // Handle paste events for telemetry
  const handlePaste = (e) => {
    if (chatId) {
      telemetryPaste(e);
    }
  };

  // Wrap sendMessage to include telemetry
  const handleSendWithTelemetry = () => {
    if (input.trim()) {
      if (chatId) {
        handleMessageSend(input); // Record telemetry before sending
      }
      sendMessage();
    }
  };

  const preprocessLatex = (text) => {
    if (!text) return text;
    
    return text
      .replace(/\\\[/g, '$$')
      .replace(/\\\]/g, '$$')
      .replace(/\\\(/g, '$')
      .replace(/\\\)/g, '$');
  };

  // ✅ NEW: Handle menu toggle with position calculation
  const handleMenuToggle = (idx, e) => {
    e.stopPropagation();
    
    if (activeMenu === idx) {
      setActiveMenu(null);
      return;
    }

    const buttonRect = e.currentTarget.getBoundingClientRect();
    setDropdownPosition({
      top: buttonRect.bottom + window.scrollY + 4,
      left: buttonRect.right + window.scrollX - 200 // 200px is dropdown width
    });
    setActiveMenu(idx);
  };

  useEffect(() => {
    const handleClickOutside = () => setActiveMenu(null);
    if (activeMenu !== null) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [activeMenu]);

  // ✅ NEW: Dropdown Portal Component
  const DropdownPortal = ({ messageIdx, messageContent }) => {
    if (activeMenu !== messageIdx) return null;

    return ReactDOM.createPortal(
      <>
        <div 
          className="dropdown-overlay"
          onClick={(e) => {
            e.stopPropagation();
            setActiveMenu(null);
          }}
        />
        <div
          className="message-dropdown"
          style={{
            position: 'fixed',
            top: `${dropdownPosition.top}px`,
            left: `${dropdownPosition.left}px`,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => handleMarkAsAnswer(messageContent)}
            disabled={submitting}
            className="dropdown-item"
          >
            <span className="dropdown-icon">📌</span>
            {submitting ? 'Submitting...' : 'Mark as Final Answer'}
          </button>
          
          <button
            onClick={() => {
              setInput(messageContent);
              setUseLatexEditor(true);
              setActiveMenu(null);
            }}
            className="dropdown-item"
          >
            <span className="dropdown-icon">📋</span>
            Copy to Editor
          </button>
        </div>
      </>,
      document.body
    );
  };

  return (
    <div className="main">
      <div className="chatbox">
        {metadata?.is_assignment_chat && (
          <div 
            className="assignment-chat-banner clickable"
            onClick={handleNavigateToAssignment}
            style={{ cursor: 'pointer' }}
            title="Click to view assignment"
          >
            <span className="banner-icon">📚</span>
            <div className="banner-info">
              <div className="banner-title">
                Q {metadata.question_number} of {metadata.assignment_title || 'Assignment'}
                {metadata.has_submitted && (
                  <span className="banner-submitted-badge">
                    ✓ Submitted (Attempt #{metadata.attempts})
                  </span>
                )}
                {metadata.is_grader_view && (
                  <span className="banner-grader-badge" style={{
                    marginLeft: '0.5rem',
                    padding: '0.25rem 0.5rem',
                    background: '#ffc107',
                    color: '#000',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: 'bold'
                  }}>
                    👁️ GRADER VIEW
                  </span>
                )}
              </div>
              <div className="banner-subtitle" style={{ fontSize: '0.85rem', color: '#666', marginTop: '2px' }}>
                Click to view assignment details →
              </div>
            </div>
          </div>
        )}

        <div className="messages">
          {messages.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              color: '#888', 
              padding: '2rem',
              fontSize: '1.1rem'
            }}>
              <p>👋 Hi! I'm ALAASKA, your AI teaching assistant.</p>
              <p>Ask me anything about your studies!</p>
            </div>
          ) : (
            <div className="messages-inner">
              {messages
                .filter(msg => msg.role !== 'system')
                .map((msg, idx) => (
                  <div 
                    key={idx} 
                    className={msg.role === 'user' ? 'user-message' : 'assistant-message'}
                  >
                    <div className="message-wrapper">
                      <div className="message-content-wrapper">
                        {msg.isStreaming && !msg.content ? (
                          <div className="typing-indicator">
                            <span className="typing-dot"></span>
                            <span className="typing-dot"></span>
                            <span className="typing-dot"></span>
                          </div>
                        ) : (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeMathjax]}
                            components={{
                              p: ({node, children, ...props}) => (
                                <p style={{ margin: '0.5rem 0', lineHeight: '1.6' }} {...props}>
                                  {children}
                                </p>
                              ),
                              ul: ({node, children, ...props}) => (
                                <ul style={{ paddingLeft: '1.5rem', margin: '0.5rem 0' }} {...props}>
                                  {children}
                                </ul>
                              ),
                              ol: ({node, children, ...props}) => (
                                <ol style={{ paddingLeft: '1.5rem', margin: '0.5rem 0' }} {...props}>
                                  {children}
                                </ol>
                              ),
                              li: ({node, children, ...props}) => (
                                <li style={{ margin: '0.25rem 0' }} {...props}>
                                  {children}
                                </li>
                              ),
                              strong: ({node, children, ...props}) => (
                                <strong style={{ fontWeight: '600' }} {...props}>
                                  {children}
                                </strong>
                              ),
                              code: ({node, inline, className, children, ...props}) => {
  // ✅ FIX: Treat single-line code without language as inline
                                const isInline = inline || (!className && String(children).trim().split('\n').length === 1);
                                
                                return isInline ? (
                                  <code 
                                    style={{
                                      background: '#f5f5f5',
                                      padding: '0.15rem 0.4rem',
                                      borderRadius: '3px',
                                      fontSize: '0.9em',
                                      fontFamily: "'Courier New', Consolas, monospace",
                                      color: '#d63384',
                                      display: 'inline'
                                    }}
                                    {...props}
                                  >
                                    {children}
                                  </code>
                                ) : (
                                  <pre 
                                    style={{
                                      background: '#2d2d2d',
                                      color: '#f8f8f2',
                                      padding: '1rem',
                                      borderRadius: '6px',
                                      overflow: 'auto',
                                      margin: '0.75rem 0'
                                    }}
                                  >
                                    <code 
                                      className={className}
                                      style={{
                                        fontFamily: "'Courier New', Consolas, monospace",
                                        fontSize: '0.9rem'
                                      }}
                                      {...props}
                                    >
                                      {children}
                                    </code>
                                  </pre>
                                );
                              }
                                                          }}
                          >
                            {preprocessLatex(msg.content)}
                          </ReactMarkdown>
                        )}
                      </div>

                      {msg.role === 'user' && metadata?.is_assignment_chat && !msg.isStreaming && (
                        <div className="message-actions">
                          <button
                            ref={el => menuButtonRefs.current[idx] = el}
                            onClick={(e) => handleMenuToggle(idx, e)}
                            disabled={submitting}
                            className={`message-menu-button ${activeMenu === idx ? 'active' : ''}`}
                            title="Options"
                          >
                            ⋮
                          </button>
                          
                          {/* ✅ CHANGED: Render dropdown via portal */}
                          <DropdownPortal messageIdx={idx} messageContent={msg.content} />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-area-wrapper">
          {useLatexEditor ? (
            <div className="latex-editor-container">
              <LatexEditor
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onSubmit={handleSendWithTelemetry}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder="Type your answer using Markdown and LaTeX..."
              />
              <div className="latex-submit-bar">
                <button 
                  onClick={() => setUseLatexEditor(false)}
                  className="latex-toggle-button-left"
                  title="Switch to Simple Editor"
                >
                  📝
                </button>
                <button 
                  onClick={handleSendWithTelemetry}
                  className="latex-send-button"
                  disabled={!input.trim()}
                >
                  Send Message (Ctrl+Enter)
                </button>
              </div>
            </div>
          ) : (
            <div className="input-container-with-toggle">
              <button
                onClick={() => setUseLatexEditor(true)}
                className="toggle-latex-button-left"
                title="Switch to LaTeX Editor for Type Theory & Formal Verification"
              >
                🔬
              </button>
              <div className="input-container">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  placeholder="Type your message here... (Shift+Enter for new line)"
                  rows="3"
                />
                <button 
                  onClick={handleSendWithTelemetry} 
                  className="send-button"
                  disabled={!input.trim()}
                >
                  ➤
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}