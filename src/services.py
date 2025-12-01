import json
import os
import asyncio
from typing import Optional, Dict, Any, List
from google.adk.runners import InMemorySessionService, InMemoryMemoryService
from google.adk.sessions.session import Session
from google.genai import types

SESSION_FILE = "data/sessions.json"
MEMORY_FILE = "data/memory.json"

class FileSessionService(InMemorySessionService):
    def __init__(self, file_path: str = SESSION_FILE):
        super().__init__()
        self.file_path = file_path
        self.load_sessions()

    def load_sessions(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    # Structure: app_name -> user_id -> session_id -> Session
                    for app_name, users in data.items():
                        if app_name not in self.sessions:
                            self.sessions[app_name] = {}
                        for user_id, sessions in users.items():
                            if user_id not in self.sessions[app_name]:
                                self.sessions[app_name][user_id] = {}
                            for session_id, session_data in sessions.items():
                                try:
                                    # Reconstruct Session object
                                    session = Session.model_validate(session_data)
                                    self.sessions[app_name][user_id][session_id] = session
                                except Exception as e:
                                    print(f"Error loading session {session_id}: {e}")
            except Exception as e:
                print(f"Error loading sessions file: {e}")

    def save_sessions(self):
        try:
            data = {}
            # self.sessions is Dict[str, Dict[str, Dict[str, Session]]]
            for app_name, users in self.sessions.items():
                data[app_name] = {}
                for user_id, sessions in users.items():
                    data[app_name][user_id] = {}
                    for session_id, session in sessions.items():
                        data[app_name][user_id][session_id] = session.model_dump(mode='json')
            
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {e}")

    async def create_session(self, *, app_name: str, user_id: str, state: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> Session:
        session = await super().create_session(app_name=app_name, user_id=user_id, state=state, session_id=session_id)
        self.save_sessions()
        return session

    async def get_session(self, *, app_name: str, user_id: str, session_id: str, config: Any = None) -> Optional[Session]:
        session = await super().get_session(app_name=app_name, user_id=user_id, session_id=session_id, config=config)
        if session:
            return session
        
        # If not found, reload from disk and try again
        print(f"Session {session_id} not found in memory, reloading from disk...")
        self.load_sessions()
        return await super().get_session(app_name=app_name, user_id=user_id, session_id=session_id, config=config)
    
    def save(self):
        self.save_sessions()


class FileMemoryService(InMemoryMemoryService):
    def __init__(self, file_path: str = MEMORY_FILE):
        super().__init__()
        self.file_path = file_path
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    # Structure: app_name/user_id -> session_id -> [events]
                    # InMemoryMemoryService stores this in self._session_events
                    if hasattr(self, "_session_events"):
                        self._session_events = data
            except Exception as e:
                print(f"Error loading memory file: {e}")

    async def add_session_to_memory(self, session: Session):
        await super().add_session_to_memory(session)
        self.save_memory()

    def save_memory(self):
        try:
            """
            self._session_events is Dict[str, Dict[str, List[Any]]]
            We need to make sure everything is serializable.
            If events are objects, we need to dump them.
            But since we can't easily iterate and check types without recursion, let's try to dump `self._session_events` directly.
            If it contains objects, json.dump will fail.
            We can use a custom encoder or Pydantic's model_dump if they are Pydantic models.
            If we assume `_session_events` contains what `add_session_to_memory` put there.
            `add_session_to_memory` likely puts `session.events`.
            `session.events` are likely Pydantic models.
            """
            def recursive_dump(obj):
                if hasattr(obj, "model_dump"):
                    return obj.model_dump(mode='json')
                if isinstance(obj, dict):
                    return {k: recursive_dump(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [recursive_dump(i) for i in obj]
                return obj

            data = recursive_dump(self._session_events)
            
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
