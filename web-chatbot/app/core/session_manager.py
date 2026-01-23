"""
Session Manager - Short-term conversation memory with TTL.

Provides session-based isolation for chat history with:
- In-memory cache (no external dependencies)
- TTL-based expiration (default 30 minutes)
- Thread-safe operations
- Message filtering (excludes greetings, acknowledgements)
- Automatic cleanup of expired sessions
"""
import uuid
import time
import threading
import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default TTL: 30 minutes in seconds
DEFAULT_TTL_SECONDS = 30 * 60

# Cleanup interval: 5 minutes
CLEANUP_INTERVAL_SECONDS = 5 * 60

# Maximum history turns to return
DEFAULT_MAX_TURNS = 10


@dataclass
class Message:
    """Single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """Chat session with messages and metadata."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()


class SessionStore:
    """
    Thread-safe in-memory session store with TTL.
    
    Features:
    - Session isolation per session_id
    - Automatic expiration of inactive sessions
    - Message filtering to exclude low-value messages
    - Window limiting to prevent context overflow
    """
    
    # Patterns for messages to filter out (greetings, acknowledgements, etc.)
    FILTER_PATTERNS = [
        # Greetings (Indonesian + English)
        r'^(hi|hai|halo|hello|hey|hei)[\s!?.,]*$',
        # Acknowledgements
        r'^(ok|oke|okay|okee|okey)[\s!?.,]*$',
        r'^(thanks|thank you|thx|ty|terima kasih|makasih|trims)[\s!?.,]*$',
        # Simple affirmatives
        r'^(ya|yes|yup|yep|iya|yaa)[\s!?.,]*$',
        # Simple negatives
        r'^(no|tidak|nope|nah|gak|ga)[\s!?.,]*$',
        # Continuations with no substance
        r'^(lanjut|next|terus|continue)[\s!?.,]*$',
    ]
    
    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        
        # Compile filter patterns once
        self._compiled_filters = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.FILTER_PATTERNS
        ]
        
        # Start background cleanup
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for cleaning expired sessions."""
        def cleanup_loop():
            while not self._shutdown.wait(CLEANUP_INTERVAL_SECONDS):
                self.cleanup_expired()
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="SessionCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Session cleanup thread started")
    
    def shutdown(self):
        """Stop the cleanup thread gracefully."""
        self._shutdown.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self._sessions[session_id] = Session(session_id=session_id)
        
        logger.debug(f"Created session: {session_id[:8]}...")
        return session_id
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists and is not expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            # Check if expired
            if time.time() - session.last_accessed > self.ttl_seconds:
                # Remove expired session
                del self._sessions[session_id]
                return False
            
            return True
    
    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """Get existing session or create new one if invalid/expired."""
        if session_id and self.session_exists(session_id):
            return session_id
        return self.create_session()
    
    def _should_filter_message(self, content: str) -> bool:
        """Check if a message should be filtered out."""
        # Filter very short messages (< 3 chars after stripping)
        stripped = content.strip()
        if len(stripped) < 3:
            return True
        
        # Check against filter patterns
        for pattern in self._compiled_filters:
            if pattern.match(stripped):
                return True
        
        return False
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a message to a session's history.
        
        Returns True if message was added, False if filtered out.
        """
        # Filter low-value messages for user messages
        # (We keep all assistant messages to maintain conversation flow)
        if role == "user" and self._should_filter_message(content):
            logger.debug(f"Filtered message in session {session_id[:8]}: {content[:20]}...")
            return False
        
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                # Session doesn't exist or expired, create new one
                session = Session(session_id=session_id)
                self._sessions[session_id] = session
            
            session.touch()
            session.messages.append(Message(role=role, content=content))
        
        return True
    
    def get_history(
        self, 
        session_id: str, 
        max_turns: int = DEFAULT_MAX_TURNS
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history for a session.
        
        Args:
            session_id: The session identifier
            max_turns: Maximum number of message pairs to return
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            
            session.touch()
            
            # Get messages, limited to max_turns * 2 (each turn = user + assistant)
            # But we take from the end, excluding the most recent user message
            # (since RAG will add it separately)
            messages = session.messages[-(max_turns * 2 + 1):-1] if session.messages else []
            
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
    
    def cleanup_expired(self) -> int:
        """
        Remove expired sessions.
        
        Returns the number of sessions removed.
        """
        current_time = time.time()
        expired_ids = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if current_time - session.last_accessed > self.ttl_seconds:
                    expired_ids.append(session_id)
            
            for session_id in expired_ids:
                del self._sessions[session_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
        
        return len(expired_ids)
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def clear_session(self, session_id: str):
        """Clear all messages from a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.clear()
                session.touch()


# Singleton instance
_session_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_session_store(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> SessionStore:
    """Get the singleton session store instance."""
    global _session_store
    
    with _store_lock:
        if _session_store is None:
            _session_store = SessionStore(ttl_seconds=ttl_seconds)
            logger.info(f"Session store initialized with TTL={ttl_seconds}s")
        return _session_store
