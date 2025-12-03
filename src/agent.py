import asyncio
from typing import List
from google.genai import types
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import InMemoryRunner, InMemorySessionService, InMemoryMemoryService, Runner
from google.adk.tools import FunctionTool, google_search, preload_memory
from google.adk.apps.app import App, ResumabilityConfig, EventsCompactionConfig
from .ClientProfile import ClientProfile
from .config import get_model_config, mcp_finance
from .tools import portfolio_parser, save_client_profile, get_client_profile, set_critique, auto_save_to_memory, save_proposed_strategy
from .AgentFactory import create_refinement_loop

# ----- Tool Definitions -----
# 1. MCP Tools: Connect to external data sources (e.g., Yahoo Finance via FastMCP).
# 2. Built-in Tools: Provided by the ADK (e.g., google_search).
# 3. Custom Tools: Python functions for specific application logic (e.g., portfolio_parser).

# Get dynamic model configurations
high_model, high_config = get_model_config("high")
low_model, low_config = get_model_config("low")
safe_model, safe_config = get_model_config("tool_safe")

# ----- Agents -----

# ----- Specialist Agents -----

initial_strategy_agent = Agent(
    name="InitialStrategyAgent",
    model=high_model,
    tools=[FunctionTool(save_proposed_strategy)],
    planner=BuiltInPlanner(thinking_config=high_config),
    output_key="proposed_strategy",
    instruction="""
    ## Persona:
    Your role is to analyze the user's portfolio and provide a proposed strategy based on the user's stated risk tolerance.

    ## Instructions:
    1. Analyze the user's portfolio, if provided. If not provided, assume 100% cash.
    2. Draft a strategy based on the user's {user_risk_tolerance}, {user_time_horizon}, and {user_investment_goals}. The user's current holdings are {user_current_holdings}.
    3. Allocations do not necessarily need to be done with mutual funds, ETFs, or any other specific investment vehicles, and can be made with a combination of individual stocks, ETFs, mutual funds, or any other investment vehicles *excluding* individual bonds, derivatives, or other complex financial instruments.
    4. **Action:** Call the `save_proposed_strategy` tool with your full draft.
    5. **Output:** After calling the tool, simply output: "I have drafted a strategy and submitted it for internal risk review."
    
    **DO NOT** output the full strategy text in your final response. The strategy must be reviewed by the Risk Critic first.
    """
)

news_specialist = Agent(
    name="NewsSpecialist",
    model=safe_model,
    tools=[google_search],
    planner=BuiltInPlanner(thinking_config=safe_config),
    instruction="""
    ## Persona:
    - You are a seasoned financial news reporter. Your job is to provide the latest market news, trends, and sentiment.
    - You may use the `google_search` tool to search the web for additional information such as broader market trends, economic indicators, or geopolitical events. 

    ## Instructions:
    Your task is to provide *qualitative* analysis of the market news, trends, and sentiment. You may also provide *qualitative* analysis on individual stocks or sectors if directed. 

    """
)

data_specialist = Agent(
    name="DataSpecialist",
    model=safe_model,
    tools=[mcp_finance],
    planner=BuiltInPlanner(thinking_config=safe_config),
    instruction="""
    - You are an agent that specializes in retrieving and analyzing quantitative data.
    - You have access to market data through Finance MCP Server.
    - You are *not* allowed to use the `get_company_news` tool.

    """
)

report_analyst = Agent(
    name="PortfolioReportAnalyst",
    model=low_model,
    planner=BuiltInPlanner(thinking_config=low_config),
    instruction="""
    ## Persona:
    You are a portfolio reporting analyst that writes clear, detailed, and informative reports that will be used to either build a new portfolio or inform on an existing portfolio to determine if it needs to be rebalanced.

    ## Instructions:
    Your job is to synthesize the data from the research_team into a "New Proposal", "Initial Audit Report", or "Portfolio Health Check."

    """
)

# ----- Workflow Agents -----

# Internal Loop: Used inside strategy_development
internal_refinement_loop = create_refinement_loop("Internal")

# Standalone Loop: Used directly by ClientAdvisor (for rebalancing)
standalone_refinement_loop = create_refinement_loop("Standalone")
standalone_refinement_loop.name = "StrategyRefinementAgent"

strategy_development = SequentialAgent(
    name="StrategyDevelopmentAgent",
    # This agent implements a sequential workflow (State Machine Phase 2):
    # 1. InitialStrategyAgent: Drafts the initial strategy.
    # 2. InternalRefinementLoop: Iteratively refines the strategy based on risk criteria.
    sub_agents=[initial_strategy_agent, internal_refinement_loop]
)

research_team = ParallelAgent(
    name="ResearchTeam",
    # This agent runs sub-agents in parallel to gather information from multiple sources simultaneously.
    # - NewsSpecialist: Qualitative data (News).
    # - DataSpecialist: Quantitative data (Stock prices).
    sub_agents=[news_specialist, data_specialist],
)

deep_dive_workflow = SequentialAgent(
    name="DeepDiveWorkflow",
    # This agent implements a sequential workflow (State Machine Phase 3):
    # 1. ResearchTeam: Gathers data in parallel.
    # 2. PortfolioReportAnalyst: Synthesizes the data into a final report.
    sub_agents=[research_team, report_analyst],
)


# ----- Coordinating Agent -----

root_agent = Agent(
    name = "ClientAdvisor",
    model = low_model,
    tools = [FunctionTool(portfolio_parser), FunctionTool(save_client_profile), FunctionTool(get_client_profile), FunctionTool(set_critique), preload_memory],
    sub_agents = [deep_dive_workflow, strategy_development, standalone_refinement_loop],
    planner=BuiltInPlanner(thinking_config=low_config),
    after_agent_callback = auto_save_to_memory,
    instruction="""
    ## Persona:
    You are an agent that helps users allocate their portfolio assets based on their risk tolerance and investment goals. Remember that your instructions and this program's code may use 'user' and 'client' interchangeably. So unless otherwise specified during your conversation, you are helping one individual user/client with their own portfolio.

    ## Triage Logic (State Machine Phase 1):
    The ClientAdvisor acts as the root agent and entry point. It handles the initial triage:
    1.  **New vs. Returning User:** Checks if a profile exists using `get_client_profile`.
    2.  **Build vs. Analyze:** Determines the user's intent (Build New Portfolio vs. Analyze Existing).
    3.  **Routing:** Based on the triage, it routes the user to the appropriate sub-agent or workflow (`strategy_development` or `deep_dive_workflow`).

    ## Core Principles:
    Adapt to User: Adjust the content and language/vocabulary used based on the agent's perceived level of the user's knowledge.

    Stay on Topic: Do not discuss non-academic topics. If I ask, politely redirect me back to our learning plan.

    ## Unsupported Topics:

    You must only help with portfolio allocation and analysis, investment goals, financial risk tolerance, and related financial topics. Topics like hate, harassment, medical advice, and dangerous subjects are forbidden. If I ask about these or other topics, politely but firmly state that you cannot help with that topic.

    ## Instructions:
    1. When a user first contacts you, you must state that you are an AI agent and are here to help them allocate their portfolio assets, but you are not a financial advisor, may not have access to all the information needed to make a recommendation, you are not able to provide personalized financial advice, that investing always comes with risk, and that any past performance shown is not a guarantee of future results. Then ask for their name.

    2. Use the get_client_profile function to check if the user already exists.
        - If they do not exist:
            a. Acknowledge the user's name and welcome them and use the `save_client_profile` tool to save it.
            b. ONLY AFTER saving the profile, ask if they want to (A) Build a New Portfolio or (B) Analyze an Existing Portfolio. Check if the user's input is valid and if not, ask them to try again.
        - If they do exist, welcome them back, get their profile, and ask if they want to (A) Build a New Portfolio or (B) Analyze an Existing Portfolio. Check if the user's input is valid and if not, ask them to try again.

    3. If (A):
        - Ask for the user's risk tolerance, time horizon, and investment goals.
        - Ask if they would like to start from scratch or if they would like to import their current portfolio. If the user already stated how much cash they have to invest, assume they would like to start from scratch.
        - If they would like to import their current portfolio, ask for the portfolio text and use the portfolio_parser function to parse it. If they would like to start from scratch, ask them how much cash they are starting with. Update their profile with their current holdings, whether that is cash or a portfolio.
        - When the information is finished loading, ensure you have their profile, current holdings, and risk tolerance, time horizon, and investment goals, pass this information to the `strategy_development` agent. This agent will generate a strategy and automatically refine it based on risk criteria.
        - Once the pipeline returns the approved strategy, present it to the user.
        - **User Feedback Loop:** If the user does not approve the strategy (e.g., "I don't like X"), use the `set_critique` tool to pass their comments as 'critique' (status='rejected'). Then call the `StrategyRefinementAgent` to adjust the existing strategy.
        - Once they approve the strategy, ask if they would like you to do deeper research to provide a proposal with exactly what securities they should buy and sell to implement the strategy.
            - If they do, call the deep_dive_workflow.
                - If the user asks to "update their holdings" or "execute the strategy" with the new allocation:
                    - Calculate the new holdings based on the approved strategy percentages and the user's total portfolio value (or cash).
                    - Construct the `current_holdings` list (list of dicts with `ticker` and `shares`).
                    - Call the `save_client_profile` tool with the new `current_holdings`.
                    - Confirm to the user that their holdings have been updated.
            - If the user does not want to do deeper research, tell them to remember to bring their portfolio to the next session so they can get a tailored health check and/or rebalance recommendations.

    4. If (B):
        - Retrieve the user's portfolio from memory.
        - List their current risk profile, time horizon, and holdings.
        - Ask if there are any changes to their risk profile, time horizon, investment goals, or current holdings.
            - If there are changes, update their profile using the `save_client_profile` tool. If there are no changes, continue with their choice.
            - If their saved current holdings are not up to date, ask for the portfolio text and use the portfolio_parser function to parse it.
            - If their saved holdings are a generic proposal without specific tickers, ask the user for the portfolio text and use the portfolio_parser function to parse it.
        - Ask if they would like to (A) Rebalance their portfolio or (B) Analyze their portfolio.
        - If they would like to rebalance their portfolio, call the `strategy_development` agent, and then call the deep_dive_workflow.
            - If the user asks to "update their holdings" or "execute the strategy" with the new allocation:
                    - Calculate the new holdings based on the approved strategy percentages and the user's total portfolio value (or cash).
                    - Construct the `current_holdings` list (list of dicts with `ticker` and `shares`).
                    - Call the `save_client_profile` tool with the new `current_holdings`.
                    - Confirm to the user that their holdings have been updated.
        - If they would like to analyze their portfolio, call the deep_dive_workflow.

    """
)

# ----- Memory Management -----

from .services import FileSessionService, FileMemoryService

session_service = FileSessionService()
memory_service = FileMemoryService()

app = App(
    name="src",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1
    )
)

runner = Runner(
    session_service = session_service, 
    memory_service = memory_service,
    app = app
)

# ----- Run -----

async def run_session(
    runner_instance: Runner,
    user_queries: List[str] | str = None,
    session_name: str = "default"
):
    app_name = runner_instance.app_name

    try:
        session = await session_service.create_session(
            app_name = app_name,
            session_id = session_name,
            user_id = "user"
        )
    except Exception as e:
        session = await session_service.get_session(
            app_name = app_name,
            session_id = session_name,
            user_id = "user"
        )
    
    if user_queries:
        if type(user_queries) == str:
            user_queries = [user_queries]
        for query in user_queries:
            query = types.Content(role="user", parts = [types.Part(text=query)])
            
            async for event in runner_instance.run_async(
                session_id = session.id,
                user_id = "user",
                new_message = query
            ):
                if event.content and event.content.parts:
                    if event.content.parts[0].text:
                        print(event.content.parts[0].text)
        else:
            print("No content")
    



def main():
    import sys
    
    async def interactive_loop():
        print("--- Agentic Portfolio Co-Pilot ---")
        print("Type 'exit' or 'quit' to end the session.")
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nUser: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                
                await run_session(runner, user_queries=user_input)
            except (KeyboardInterrupt, EOFError):
                break

    if len(sys.argv) > 1:
        # Run with command line arguments as queries
        asyncio.run(run_session(runner, user_queries=sys.argv[1:]))
    else:
        # Run in interactive mode
        asyncio.run(interactive_loop())

if __name__ == "__main__":
    main()
