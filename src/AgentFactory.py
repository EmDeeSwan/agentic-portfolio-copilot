from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import FunctionTool, google_search
from .config import get_model_config
from .tools import exit_loop, save_proposed_strategy
from google.adk.agents.callback_context import CallbackContext

def should_continue_loop(ctx: CallbackContext) -> bool:
    """
    Returns True if the loop should continue, False if it should stop.
    """
    return not ctx.state.get("loop_complete", False)


def create_refinement_loop(suffix: str) -> LoopAgent:
    """
    Creating a factory for the refinement loop because there can be two subtly different refinement loops.
    1. The initial strategy refinement loop
    2. The user rejects the initial strategy and requires a modified version based on their comments, which should bypass the initial agent generator step.
    """
    
    critic_model, critic_config = get_model_config("high")

    critic = Agent(
        name=f"RiskCriticAgent_{suffix}",
        model=critic_model,
        planner=BuiltInPlanner(thinking_config=critic_config),
        output_key="critique",
        tools = [google_search],
        instruction="""
        ## Persona:
        Your role is to analyze the risk strategy proposed to the user's stated risk tolerance by the PortfolioStrategyAgent and either approve it or reject it.

        ## Instructions:
        1. Analyze the risk strategy proposed to the user's stated risk tolerance by the PortfolioStrategyAgent.
        2. If you approve it, you *MUST* return the 'approved' status.
        3. If you reject it, you *MUST* return the 'rejected' status with a reason.

        The user's risk tolerance is {user_risk_tolerance}.
        The user's time horizon is {user_time_horizon}.
        The user's investment goals are {user_investment_goals}.
        
        The proposed strategy is {proposed_strategy}.

        ## Allocation Examples

        The following tables are examples of portfolio allocations by risk tolerance. They are not all encompassing, but they are a good starting point. Allocations do not necessarily need to be done with mutual funds, ETFs, or any other specific investment vehicles, and can be made with a combination of individual stocks, ETFs, mutual funds, or any other investment vehicles *excluding* individual bonds, derivatives, or other complex financial instruments.

        ### Conservative Portfolio

        | Feature | Vanguard LifeStrategy Income (VASIX) | Fidelity Asset Manager 20% (FASIX) | Schwab MarketTrack Conservative (SWCGX) |
        | :--- | :--- | :--- | :--- |
        | **Equity Allocation** | 20% | 20% | ~40% |
        | **Bond Allocation** | 80% | 50% | ~55% |
        | **Cash/Short-Term** | ~0% (Fully Invested) | 30% (High Liquidity) | ~5% |
        | **Management Style** | Passive (Index Fund of Funds) | Active (Tactical Shifts) | Hybrid (Active/Passive) |
        | **Int'l Equity Bias** | High (~40% of Equity) | Moderate | Moderate |
        | **Primary Goal** | Income & Stability | Capital Preservation & Liquidity | Moderate Stability & Growth |

        ### Moderate Portfolio

        | Feature                | Vanguard Moderate (VSMGX) | BlackRock Target 60/40 | Fidelity Asset Manager 60% (FSANX) | Schwab MarketTrack Balanced (SWBGX) |
        | :--------------------- | :------------------------ | :--------------------- | :--------------------------------- | :---------------------------------- |
        | **Philosophy**         | Passive / Strategic       | ETF / Tactical         | Active / Tactical                  | Hybrid                              |
        | **Equity/Bond/Cash**   | 60% / 40% / 0%            | 60% / 40% / 0%         | 60% / 35% / 5%                     | 60% / 35% / 5%                      |
        | **Int'l Equity Bias**  | High (~40% of Equity)     | Moderate               | Moderate (~45% of Equity)          | Low (~30% of Equity)                |
        | **Rebalancing**        | Continuous/Static         | Dynamic                | Discretionary                      | Automatic                           |

        ### Growth Portfolio

        | Metric                  | Vanguard LifeStrategy Growth (VASGX) | iShares Core Aggressive Allocation (AOA) | Fidelity Asset Manager 85% (FAMRX)          | Schwab MarketTrack Growth (SWHGX)            |
        | :---------------------- | :----------------------------------- | :--------------------------------------- | :------------------------------------------ | :------------------------------------------- |
        | **Structure**           | Mutual Fund (Fund of Funds)          | ETF (Fund of Funds)                      | Mutual Fund (Active)                        | Mutual Fund (Hybrid)                         |
        | **Objective**           | Growth of capital & some income      | Track S&P Target Risk Aggressive index   | Maximize total return via active allocation | High capital growth via fundamental indexing |
        | **Total Equity Alloc.** | 81.38%                               | ~89.92%**                                | 88.66% (50.98% Dom + 37.68% Int)            | 79.03% (52.45% Dom + 26.54% Int)             |
        | **Domestic Equity**     | 48.90%                               | 45.93%                                   | 50.98%                                      | 52.45%                                       |
        | **Int'l Equity**        | 33.00%                               | 43.99% (Dev + EM)                        | 37.68%                                      | 26.54%                                       |
        | **Fixed Income Alloc.** | 18.03%                               | 19.95%                                   | 13.70%                                      | 15.80%                                       |
        | **Cash / Other**        | 0.59%                                | 0.14%                                    | -2.59% (Net Other Assets)                   | 4.5%                                         |
        | **Portfolio Turnover**  | Low (Rebalancing only)               | Low (Rebalancing only)                   | 42.00%                                      | 17.00%                                       |
        

        """
    )

    refiner_model, refiner_config = get_model_config("high")

    refiner = Agent(
        name=f"PortfolioStrategyRefinementAgent_{suffix}",
        model=refiner_model,
        tools=[FunctionTool(exit_loop), FunctionTool(save_proposed_strategy)],
        planner=BuiltInPlanner(thinking_config=refiner_config),
        output_key="proposed_strategy",
        instruction="""
        ## Persona:
        Your role is to refine the investment strategy proposed to the client. You have the proposal and critique in order to make adjustments.

        Proposal: {proposed_strategy}
        Critique: {critique}

        ## Instructions:
        1. **Check for User Feedback FIRST**: Look at the conversation history. If the user has explicitly asked for a change (e.g., "I don't like this", "Change X") since the last strategy was generated, you *MUST* refine the strategy based on their request. In this case, IGNORE any previous "approved" status from the critic and DO NOT call exit_loop yet.
        
        2. **Processing**:
            - IF the critique is *EXACTLY* "approved" (and there is NO new user feedback): 
                **ACTION:** You MUST output the full text of the APPROVED strategy now (so the user can see it).
                **THEN:** Call the 'exit_loop' function immediately.
            
            - IF the critique starts with "rejected" OR you are addressing user feedback:
                **ACTION:** Rewrite the proposed strategy to specifically address the critique points or user feedback.
                **ACTION:** Call the `save_proposed_strategy` tool with your new draft.
                **OUTPUT:** A brief message stating you have updated the strategy and submitted it for review (e.g., "I have updated the strategy based on feedback.").
            
            - IF there is NO critique (empty) and NO user feedback:
                **ACTION:** Do NOT output the strategy.
                **OUTPUT:** simply state "Sending strategy for risk review."
        """
    )

    sequence = SequentialAgent(
        name=f"RefinementSequence_{suffix}",
        sub_agents=[refiner, critic]
    )

    return LoopAgent(
        name=f"StrategyRefinementLoop_{suffix}",
        sub_agents=[sequence],
        max_iterations=5,                
    )