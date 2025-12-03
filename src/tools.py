import json
import re
import os
from typing import Dict, Any, Optional
from google.adk.tools import ToolContext
from google.adk.agents.callback_context import CallbackContext
from .ClientProfile import ClientProfile

PROFILES_FILE = "data/user_profiles.json"

def _load_profiles() -> Dict[str, Any]:
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_profiles(profiles: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(PROFILES_FILE), exist_ok=True)
        with open(PROFILES_FILE, "w") as f:
            json.dump(profiles, f, indent=2)
    except Exception as e:
        print(f"Error saving profiles: {e}")

# ----- Portfolio Tools -----

def portfolio_parser(portfolio_text: str) -> str:
    """
    Parses unstructured portfolio text (e.g., '10 VTI @ $200') into JSON.
    This tool will be given to the ClientAdvisor.
    """
    print(f"\n[Tool: PortfolioParser] Processing: '{portfolio_text}'...")
    holdings = []
    # Regex handles: "10 VTI", "10 shares of VTI", "10 VTI @ 250", "10 VTI at $250.50"
    matches = re.findall(r"(\d+)\s+(?:shares\s+of\s+)?([A-Za-z]+)(?:\s*(?:@|at)\s*\$?([\d,.]+))?", portfolio_text, re.IGNORECASE)
    
    for qty, ticker, cost in matches:
        entry = {"ticker": ticker.upper(), "shares": int(qty)}
        if cost:
            try:
                entry["cost_basis"] = float(cost.replace(",", ""))
            except ValueError:
                pass 
        holdings.append(entry)
    
    if not holdings:
        return "Error: No valid holdings found. Format example: '10 VTI @ $200, 5 AAPL'"
    
    return f"Success. Parsed and saved: {json.dumps(holdings)}"

# ----- Profile Tools -----

def save_client_profile(tool_context: ToolContext, client_profile: ClientProfile) -> Dict[str, Any]:
    """
    Saves the client profile to the database (json file).
    """
    # Handle case where client_profile is passed as a dict
    if isinstance(client_profile, dict):
        client_profile = ClientProfile(**client_profile)

    # Resolve name: use provided name or fallback to session state
    profile_name = client_profile.name
    if not profile_name:
        profile_name = tool_context.state.get("user_name")

    # Update session state
    if client_profile.name:
        tool_context.state["user_name"] = client_profile.name
    if client_profile.risk_tolerance:
        tool_context.state["user_risk_tolerance"] = client_profile.risk_tolerance
    if client_profile.time_horizon:
        tool_context.state["user_time_horizon"] = client_profile.time_horizon
    if client_profile.investment_goals:
        tool_context.state["user_investment_goals"] = client_profile.investment_goals
    if client_profile.current_holdings:
        tool_context.state["user_current_holdings"] = client_profile.current_holdings
    
    tool_context.state["critique"] = "None"

    # Save to json file
    if profile_name:
        profiles = _load_profiles()
        
        # Convert ClientProfile to dict for storage
        new_profile_data = {}
        if hasattr(client_profile, "model_dump"):
            new_profile_data = client_profile.model_dump(mode='json', exclude_unset=True)
        elif hasattr(client_profile, "dict"):
            new_profile_data = client_profile.dict(exclude_unset=True)
        else:
            new_profile_data = {k: v for k, v in client_profile.__dict__.items() if v is not None}
            
        # Merge with existing profile if it exists
        if profile_name in profiles:
            existing_profile = profiles[profile_name]
            existing_profile.update(new_profile_data)
            profiles[profile_name] = existing_profile
        else:
            # Ensure name is present for new profiles
            if "name" not in new_profile_data:
                new_profile_data["name"] = profile_name
            profiles[profile_name] = new_profile_data
            
        _save_profiles(profiles)

    return {"status": "success", "message": "Client profile saved successfully."}

def get_client_profile(tool_context: ToolContext, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the client profile. 
    If 'name' is provided, it looks up the profile in json file.
    If 'name' is NOT provided, it returns the profile currently saved in the session context.
    """
    # 1. Try to get from session state first if no name provided
    if not name:
        name = tool_context.state.get("user_name")
    
    if not name:
        return {"status": "not_found", "message": "Client profile not found. Please provide a name or create a new profile."}

    # 2. Try to load from json file
    profiles = _load_profiles()
    if name in profiles:
        profile_data = profiles[name]
        
        # Populate session state with loaded data so subsequent calls work
        tool_context.state["user_name"] = profile_data.get("name")
        tool_context.state["user_risk_tolerance"] = profile_data.get("risk_tolerance")
        tool_context.state["user_time_horizon"] = profile_data.get("time_horizon")
        tool_context.state["user_investment_goals"] = profile_data.get("investment_goals")
        tool_context.state["user_current_holdings"] = profile_data.get("current_holdings")
        
        return profile_data

    # 3. If not in json file but in session (partial data?), return session data
    if tool_context.state.get("user_name") == name:
        return {
            "name": name, 
            "risk_tolerance": tool_context.state.get("user_risk_tolerance"), 
            "time_horizon": tool_context.state.get("user_time_horizon"), 
            "investment_goals": tool_context.state.get("user_investment_goals"), 
            "current_holdings": tool_context.state.get("user_current_holdings")
        }

    return {"status": "not_found", "message": f"Client profile for '{name}' not found."}

# ----- Workflow/Loop Tools -----

def save_proposed_strategy(tool_context: ToolContext, strategy_content: str) -> Dict[str, Any]:
    """
    Saves the drafted strategy to the session state so the Critic can review it.
    This allows the agent to draft the strategy silently without outputting it to the user chat immediately.
    """
    tool_context.state["proposed_strategy"] = strategy_content
    # Reset the critique since we have a new proposal
    tool_context.state["critique"] = "None"
    return {"status": "success", "message": "Strategy saved to state. Ready for critique."}

def exit_loop(tool_context: ToolContext):
    """
    Call this function ONLY when the critique is 'APPROVED', indicating that the strategy is sound and should be implemented.
    """
    tool_context.actions.escalate = True
    return {"status": "approved", "message": "Strategy approved. Exiting loop."}

def rerun_loop(tool_context: ToolContext, reason: str):
    """
    Call this function ONLY when the critique is 'REJECTED', indicating that the strategy is not sound and should not be implemented.
    """
    tool_context.state["loop_complete"] = False
    return {"status": "rejected", "message": f"Strategy rejected. Improve the strategy by {reason}."}

def set_critique(tool_context: ToolContext, critique: str, status: str = "rejected") -> Dict[str, Any]:
    """
    Sets the critique for the strategy refinement agent.
    """
    tool_context.state["critique"] = f"{status}: {critique}"
    tool_context.state["loop_complete"] = False
    return {"status": "success", "message": "Critique set successfully."}

async def auto_save_to_memory(callback_context: CallbackContext):
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )