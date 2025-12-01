import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.adk.tools import FunctionTool, ToolContext, McpToolset
from google.adk.apps.app import App, ResumabilityConfig
from mcp.client.stdio import StdioServerParameters 

# Load environment variables
base_dir = Path(__file__).resolve().parent
if (base_dir / ".env").exists():
    load_dotenv(base_dir / ".env")
elif (base_dir.parent / ".env").exists():
    load_dotenv(base_dir.parent / ".env")
else:
    load_dotenv()

# 1. Setup Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# 2. Model Setup
client = genai.Client(api_key=GOOGLE_API_KEY)

low_thinking_config = types.ThinkingConfig(
    thinking_level="low",
    include_thoughts=True
)

high_thinking_config = types.ThinkingConfig(
    thinking_level="high",
    include_thoughts=True
)

tool_safe_config = types.ThinkingConfig(
    include_thoughts=False
)

PRIMARY_MODEL = "gemini-3-pro-preview"
FALLBACK_HIGH = "gemini-2.5-pro"
FALLBACK_LOW = "gemini-2.5-flash"

_model_availability_cache = {}

def is_model_available(model_name: str) -> bool:
    if model_name in _model_availability_cache:
        return _model_availability_cache[model_name]
    
    try:
        # Check if model works by generating a small token.
        # This catches 429s and other runtime errors.
        client.models.generate_content(
            model=model_name,
            contents="test"
        )
        _model_availability_cache[model_name] = True
        return True
    except Exception as e:
        print(f"Model {model_name} check failed: {e}")
        _model_availability_cache[model_name] = False
        return False

def get_model_config(thinking_level: str = "low"):
    """
    Returns (model_name, thinking_config) based on availability and requested level.
    """
    # Check primary availability
    if is_model_available(PRIMARY_MODEL):
        if thinking_level == "high":
            return PRIMARY_MODEL, high_thinking_config
        elif thinking_level == "low":
            return PRIMARY_MODEL, low_thinking_config
        else: # tool_safe or others
            return PRIMARY_MODEL, tool_safe_config
    else:
        # Fallback logic
        if thinking_level == "high":
            return FALLBACK_HIGH, tool_safe_config
        elif thinking_level == "low":
            return FALLBACK_LOW, tool_safe_config
        else:
            return FALLBACK_LOW, tool_safe_config

# Keeping usedModel for backward compatibility if needed, but we will update usages with get_model_config() for the newer fallback logic.
usedModel = PRIMARY_MODEL

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# 3. MCP Setup (Yahoo Finance Local Server)

server_script_path = str(base_dir / "finance_server.py")

mcp_finance = McpToolset(
    connection_params=StdioServerParameters(
        command=sys.executable, 
        args=[server_script_path],
        env=os.environ.copy() # No special API keys needed!
    ),
    tool_filter=None 
)