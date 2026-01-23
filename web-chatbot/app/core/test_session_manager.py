"""
Unit tests for Session Manager.

Tests:
- Session creation and retrieval
- Message storage and filtering
- TTL expiration
- Session isolation
- Window limiting
"""
import pytest
import time
from unittest.mock import patch
from app.core.session_manager import (
    SessionStore, 
    get_session_store,
    Message,
    Session,
    DEFAULT_TTL_SECONDS,
    DEFAULT_MAX_TURNS
)


class TestSessionStore:
    """Tests for SessionStore class."""
    
    def setup_method(self):
        """Create a fresh SessionStore for each test."""
        # Use short TTL for testing
        self.store = SessionStore(ttl_seconds=2)
    
    def teardown_method(self):
        """Cleanup after each test."""
        self.store.shutdown()
    
    def test_create_session_returns_valid_uuid(self):
        """Test session creation returns a valid UUID4 string."""
        session_id = self.store.create_session()
        
        assert session_id is not None
        assert len(session_id) == 36  # UUID format: 8-4-4-4-12
        assert '-' in session_id
    
    def test_session_exists_returns_true_for_valid_session(self):
        """Test session_exists returns True for newly created session."""
        session_id = self.store.create_session()
        
        assert self.store.session_exists(session_id) is True
    
    def test_session_exists_returns_false_for_invalid_session(self):
        """Test session_exists returns False for non-existent session."""
        assert self.store.session_exists("invalid-session-id") is False
    
    def test_add_and_get_messages(self):
        """Test adding and retrieving messages."""
        session_id = self.store.create_session()
        
        self.store.add_message(session_id, "user", "What is BBRI?")
        self.store.add_message(session_id, "assistant", "BBRI is Bank Rakyat Indonesia.")
        self.store.add_message(session_id, "user", "What is its market cap?")
        
        history = self.store.get_history(session_id, max_turns=10)
        
        # Should return first 2 messages (excluding last user message)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is BBRI?"
        assert history[1]["role"] == "assistant"
    
    def test_message_filtering_greetings(self):
        """Test that greetings are filtered out."""
        session_id = self.store.create_session()
        
        # These should be filtered
        result1 = self.store.add_message(session_id, "user", "hi")
        result2 = self.store.add_message(session_id, "user", "hello!")
        result3 = self.store.add_message(session_id, "user", "halo")
        
        assert result1 is False
        assert result2 is False
        assert result3 is False
        
        # Meaningful message should not be filtered
        result4 = self.store.add_message(session_id, "user", "What is BBRI stock price?")
        assert result4 is True
    
    def test_message_filtering_acknowledgements(self):
        """Test that acknowledgements are filtered out."""
        session_id = self.store.create_session()
        
        # These should be filtered
        assert self.store.add_message(session_id, "user", "ok") is False
        assert self.store.add_message(session_id, "user", "thanks") is False
        assert self.store.add_message(session_id, "user", "terima kasih") is False
    
    def test_message_filtering_short_messages(self):
        """Test that very short messages are filtered out."""
        session_id = self.store.create_session()
        
        assert self.store.add_message(session_id, "user", "ab") is False
        assert self.store.add_message(session_id, "user", "y") is False
    
    def test_assistant_messages_not_filtered(self):
        """Test that assistant messages are never filtered."""
        session_id = self.store.create_session()
        
        # Even short assistant messages should be kept
        assert self.store.add_message(session_id, "assistant", "ok") is True
        assert self.store.add_message(session_id, "assistant", "hi") is True
    
    def test_session_isolation(self):
        """Test that sessions are isolated from each other."""
        session_1 = self.store.create_session()
        session_2 = self.store.create_session()
        
        self.store.add_message(session_1, "user", "Query for session 1")
        self.store.add_message(session_1, "assistant", "Response for session 1")
        
        self.store.add_message(session_2, "user", "Query for session 2")
        self.store.add_message(session_2, "assistant", "Response for session 2")
        
        history_1 = self.store.get_history(session_1)
        history_2 = self.store.get_history(session_2)
        
        # Each session should only see its own messages
        assert len(history_1) == 1  # Excludes last user message
        assert "session 1" in history_1[0]["content"]
        
        assert len(history_2) == 1
        assert "session 2" in history_2[0]["content"]
    
    def test_window_limiting(self):
        """Test that history is limited to max_turns."""
        session_id = self.store.create_session()
        
        # Add many messages
        for i in range(20):
            self.store.add_message(session_id, "user", f"User message {i}")
            self.store.add_message(session_id, "assistant", f"Assistant message {i}")
        
        # Get limited history
        history = self.store.get_history(session_id, max_turns=3)
        
        # Should return max_turns * 2 messages (not including last user message we add)
        assert len(history) <= 7  # 3 turns * 2 + 1 (buffer)
    
    def test_ttl_expiration(self):
        """Test that sessions expire after TTL."""
        session_id = self.store.create_session()
        self.store.add_message(session_id, "user", "Test message")
        
        assert self.store.session_exists(session_id) is True
        
        # Wait for TTL to expire
        time.sleep(3)  # TTL is 2 seconds in test
        
        assert self.store.session_exists(session_id) is False
    
    def test_get_or_create_session_creates_new_for_invalid(self):
        """Test get_or_create_session creates new session for invalid ID."""
        new_id = self.store.get_or_create_session("invalid-id")
        
        assert new_id != "invalid-id"
        assert self.store.session_exists(new_id) is True
    
    def test_get_or_create_session_returns_existing_for_valid(self):
        """Test get_or_create_session returns existing session for valid ID."""
        existing_id = self.store.create_session()
        returned_id = self.store.get_or_create_session(existing_id)
        
        assert returned_id == existing_id
    
    def test_cleanup_expired_removes_stale_sessions(self):
        """Test that cleanup_expired removes stale sessions."""
        session_1 = self.store.create_session()
        session_2 = self.store.create_session()
        
        assert self.store.get_session_count() == 2
        
        # Wait for TTL
        time.sleep(3)
        
        removed = self.store.cleanup_expired()
        
        assert removed == 2
        assert self.store.get_session_count() == 0
    
    def test_clear_session(self):
        """Test clearing a session's messages."""
        session_id = self.store.create_session()
        self.store.add_message(session_id, "user", "Test message")
        self.store.add_message(session_id, "assistant", "Response")
        
        self.store.clear_session(session_id)
        
        history = self.store.get_history(session_id)
        assert len(history) == 0


class TestSingleton:
    """Test get_session_store singleton behavior."""
    
    def test_returns_same_instance(self):
        """Test that get_session_store returns the same instance."""
        # Note: This may fail if run after other tests due to global state
        # In real usage, the singleton is created once at app startup
        store1 = get_session_store()
        store2 = get_session_store()
        
        assert store1 is store2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
