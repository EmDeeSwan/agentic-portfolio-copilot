from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class ClientProfile(BaseModel):
    name: Optional[str] = Field(default=None, description="The client's name")
    risk_tolerance: Optional[str] = Field(default=None, description="The client's risk tolerance. eg: 'Conservative', 'Moderate', 'Aggressive'")
    time_horizon: Optional[str] = Field(default=None, description="The client's time horizon. eg: '10 years'")
    investment_goals: Optional[str] = Field(default=None, description="The client's investment goals. eg: 'Retirement', 'Buy a house', 'Pay off debt', 'Build an emergency fund'")
    current_holdings: Optional[List[Dict[str, Any]]] = Field(default=None, description="The client's current holdings")