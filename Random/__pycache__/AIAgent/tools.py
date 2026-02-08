from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from datetime import datetime
import math
import re

from langchain_core.load import serializable


# Create the search tool using the DuckDuckGo search function directly
search_tool = DuckDuckGoSearchRun()

# Create the Wikipedia tool for querying Wikipedia articles
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def calculator(expression: str) -> str:
    """
    Performs mathematical calculations and evaluates arithmetic expressions.
    Supports basic operations (+, -, *, /), exponents (**), and common math functions.
    Examples: '67 * 67', '2 ** 10', 'sqrt(144)', 'sin(3.14)', '(5 + 3) * 2'
    """
    try:
        # Create a safe namespace with math functions
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        # Clean the expression (remove any potentially dangerous characters)
        if re.search(r'[a-zA-Z_]', expression):
            # If it contains letters, only allow if they're math functions
            if not all(word in safe_dict or word.isdigit() for word in re.findall(r'[a-zA-Z_]+', expression)):
                return "Error: Invalid expression. Only basic arithmetic and math functions are allowed."
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"The result is: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"