# Agentic Portfolio Co-Pilot

## Problem
Retail investors often struggle to manage their portfolios effectively due to a lack of personalized advice, difficulty in keeping up with market news, and the complexity of rebalancing assets. Traditional robo-advisors can be rigid, while human advisors are expensive. There is a need for an intelligent, interactive system that can understand a user's specific financial goals, risk tolerance, and current holdings to provide tailored investment strategies and real-time market insights.

## Concept & Innovation
The **Agentic Portfolio Co-Pilot** is a stateful, multi-agent system designed to manage a retail investor's portfolio. Unlike simple chatbots, this system acts as a persistent "Co-Pilot" that evolves with the user.

### Central Idea & Value
The core innovation is the use of **specialized agents** to solve the "blank page" problem for new investors and the "black box" problem for existing ones.
-   **For New Investors:** It acts as a *Creator*, drafting personalized strategies from scratch.
-   **For Existing Investors:** It acts as an *Auditor*, parsing unstructured portfolio data to provide health checks.
-   **For Everyone:** It provides ongoing *Maintenance*, remembering user context across sessions to offer proactive rebalancing and market analysis.

### Why Agents?
Agents are central to this solution because they enable:
-   **Stateful Triage:** The system intelligently routes users to "Build" or "Audit" workflows based on their profile history.
-   **Parallel Research:** Multiple specialists (News, Data) work simultaneously to gather comprehensive market intelligence.
-   **Human-in-the-Loop:** The user remains the decision-maker, approving strategies before any "execution" logic is simulated.

## Implementation Architecture

The system is built on a modular architecture using **Google's Agent Development Kit (ADK)** and **Gemini 3**, but will fall back to **Gemini 2.5** if Gemini 3 is not available.

### Technical Design
The application orchestrates a team of agents using ADK's `SequentialAgent` and `ParallelAgent` patterns:

1.  **ClientAdvisor (Root Agent):** The "Manager" that handles user intake and triage. It uses a **custom tool** (`portfolio_parser`) to convert unstructured text (e.g., "10 shares of NVDA") into structured JSON data.
2.  **StrategyDevelopment (Sequential Workflow):**
    -   `InitialStrategyAgent`: Drafts a proposal based on risk tolerance.
    -   **Refinement Loop:** A feedback loop where a `StrategyRefinementAgent` critiques and improves the strategy until it meets quality standards.
3.  **DeepDiveWorkflow (Research Engine):**
    -   **ResearchTeam (Parallel Agent):** Runs `NewsSpecialist` (Google Search) and `DataSpecialist` (Yahoo Finance MCP) concurrently to maximize efficiency.
    -   `PortfolioReportAnalyst`: Synthesizes the multi-modal data into a final report.

### AI Integration
-   **Gemini 3 Pro Preview:** Used for complex reasoning tasks (Strategy, Analysis).
-   **Thinking Configs:** We leverage `include_thoughts=True` to expose the model's hidden reasoning ("thought signatures"), allowing for transparent debugging of the agent's decision-making process.
-   **Dynamic Fallback:** The system automatically falls back to `Gemini 2.5 Pro` or `Flash` if the primary model is unavailable, ensuring high availability.

## Setup & Demo Guide

### Prerequisites
-   Python 3.10+
-   Google Cloud Project with Gemini API enabled.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd agentic-portfolio-copilot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    -   Create a `.env` file in the root directory.
    -   Add your Google API key:
        ```env
        GOOGLE_API_KEY=your_google_api_key
        ```

### How to Demo (Running the Application)

This project uses the Google ADK CLI to serve a web-based chat interface.

1.  **Start the Server:**
    ```bash
    adk web --port 8000
    ```

2.  **Open the UI:**
    Navigate to `http://localhost:8000` in your browser.

3.  **Demo Scenarios:**

    *   **Scenario A: The New Investor (Build)**
        1.  Enter a new name (e.g., "Alice").
        2.  The agent will ask for your risk tolerance. Reply: "High risk, long term."
        3.  Choose to **Build a New Portfolio**.
        4.  Watch the agent draft a strategy, refine it, and present a proposal.

    *   **Scenario B: The Existing Investor (Audit)**
        1.  Enter a new name (e.g., "Bob").
        2.  Provide your profile info.
        3.  Choose to **Analyze an Existing Portfolio**.
        4.  Paste a portfolio: "I have 10 shares of AAPL and 5 shares of GOOGL."
        5.  Observe the `ResearchTeam` gather live data and news to analyze your holdings.
