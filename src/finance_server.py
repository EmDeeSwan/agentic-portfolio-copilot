import yfinance as yf
from mcp.server.fastmcp import FastMCP
import json

# Initialize a simple MCP server
mcp = FastMCP("YahooFinance")

@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    Get the current price and basic info for a stock symbol (e.g., AAPL, NVDA).
    Returns a JSON string with price, currency, and day's change.
    """
    try:
        ticker = yf.Ticker(symbol)
        # fast_info is often faster than .info for real-time price
        price = ticker.fast_info.last_price
        prev_close = ticker.fast_info.previous_close
        change = price - prev_close
        percent = (change / prev_close) * 100
        
        return json.dumps({
            "symbol": symbol.upper(),
            "price": round(price, 2),
            "change": round(change, 2),
            "percent_change": f"{round(percent, 2)}%",
            "currency": ticker.fast_info.currency
        })
    except Exception as e:
        return f"Error fetching price for {symbol}: {str(e)}"

@mcp.tool()
def get_company_news(symbol: str) -> str:
    """
    Get the latest news headlines for a company.
    Returns a list of titles and links.
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return f"No recent news found for {symbol}."
        
        # Format neatly
        formatted_news = []
        for item in news[:5]: # Top 5 stories
            formatted_news.append(f"- {item.get('title')} ({item.get('publisher')})")
        
        return "\n".join(formatted_news)
    except Exception as e:
        return f"Error fetching news for {symbol}: {str(e)}"

@mcp.tool()
def get_company_info(symbol: str) -> str:
    """
    Get fundamental company information (sector, business summary, PE ratio, market cap).
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key fields to avoid overwhelming the context
        key_data = {
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "summary": info.get("longBusinessSummary", "")[:500] + "..." # Truncate summary
        }
        return json.dumps(key_data, indent=2)
    except Exception as e:
        return f"Error fetching info for {symbol}: {str(e)}"

@mcp.tool()
def get_analyst_recommendations(symbol: str) -> str:
    """
    Get recent analyst recommendations and target prices for a stock.
    """
    try:
        ticker = yf.Ticker(symbol)
        # This returns a pandas DataFrame, so we convert to string
        recs = ticker.recommendations
        if recs is None or recs.empty:
            return "No recommendations available."
        
        # Get latest 5
        return recs.tail(5).to_string()
    except Exception as e:
        return f"Error fetching recommendations: {str(e)}"

if __name__ == "__main__":
    # This runs the server over stdio (standard input/output)
    mcp.run()