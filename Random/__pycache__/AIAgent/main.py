from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import search_tool, wikipedia_tool, calculator

load_dotenv()

# Define the structured output format for research responses
class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic or subject being researched")
    summary: str = Field(description="A comprehensive summary of the key findings and information, 2-4 sentences")
    sources: list[str] = Field(description="List of URLs, references, or source names that were consulted")
    tools_used: list[str] = Field(description="Names of tools, methods, or knowledge bases used to gather the information")


llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools([search_tool, wikipedia_tool, calculator])

def run_agent_with_tools(query: str):
    """
    Simple agent loop that can use tools and return structured output
    """
    messages = [
        SystemMessage(content="You are a helpful research assistant. Use the available tools to search for information when needed."),
        HumanMessage(content=query)
    ]
    
    # Step 1: Call LLM with tools
    response = llm_with_tools.invoke(messages)
    
    # Step 2: Check if LLM wants to use tools
    if response.tool_calls:
        print(f"🔧 Agent is using {len(response.tool_calls)} tool(s)...")
        
        # Add AI response to messages
        messages.append(response)
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            print(f"   Calling tool: {tool_name} with input: {tool_input}")
            
            # Execute the tool based on tool name
            if tool_name == "duckduckgo_search":
                if hasattr(search_tool, 'invoke'):
                    tool_result = search_tool.invoke(tool_input)
                else:
                    # Fallback for older LangChain versions
                    tool_result = search_tool.run(tool_input.get("query", ""))
            elif tool_name == "wikipedia":
                if hasattr(wikipedia_tool, 'invoke'):
                    tool_result = wikipedia_tool.invoke(tool_input)
                else:
                    # Fallback for older LangChain versions
                    tool_result = wikipedia_tool.run(tool_input.get("query", ""))
            elif tool_name == "calculator":
                if hasattr(calculator, 'invoke'):
                    tool_result = calculator.invoke(tool_input)
                else:
                    # Fallback for older LangChain versions
                    tool_result = calculator.run(tool_input.get("expression", ""))
            else:
                tool_result = f"Unknown tool: {tool_name}"
            
            print(f"   Tool result preview: {tool_result[:200]}...")
            
            # Add tool result to messages using ToolMessage
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
        
        # Step 3: Call LLM again with tool results to get final answer
        response = llm_with_tools.invoke(messages)
    
    # Step 4: Get structured output
    final_content = response.content
    
    # Ask LLM to format the response in structured format
    llm_with_structure = llm.with_structured_output(ResearchResponse)
    structured_response = llm_with_structure.invoke([
        SystemMessage(content="Format the following information into the structured format."),
        HumanMessage(content=f"Question: {query}\n\nInformation gathered: {final_content}")
    ])
    
    return structured_response

# Test the agent with tools
query = "what is 20*(10+5)/7 in fraction form?"
print(f"Query: {query}\n")

try:
    response = run_agent_with_tools(query)
    print(f"\n✅ Results:")
    print(f"Topic: {response.topic}")
    print(f"Summary: {response.summary}")
    print(f"Sources: {response.sources}")
    print(f"Tools used: {response.tools_used}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
