"""A specialist agent that builds an itinerary using Tavily search."""

import json
import os
from urllib.request import Request, urlopen

from braintrust import BraintrustState, init_logger
from openai import OpenAI


def create_logger():
    api_key = os.getenv("BRAINTRUST_PROJECT_C_API_KEY")
    if not api_key:
        raise RuntimeError("Set BRAINTRUST_PROJECT_C_API_KEY in your environment.")
    return init_logger(
        project=os.getenv("BRAINTRUST_PROJECT_C_NAME", "Project C"),
        api_key=api_key,
        # Keep this project's API key separate from the other logger states.
        state=BraintrustState(),
        set_current=False,
    )


def search_web(query: str, agent_span, session_id: str) -> str:
    """Search the web with Tavily and return a few useful results."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Set TAVILY_API_KEY in your environment before running the demo.")

    request = Request(
        "https://api.tavily.com/search",
        data=json.dumps({"api_key": api_key, "query": query, "max_results": 5}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with agent_span.start_span(name="tavily_search", type="tool", set_current=False) as span:
        span.log(input={"query": query}, metadata={"SessionId": session_id})
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read())
        results = json.dumps(data.get("results", []))
        span.log(output=results, metadata={"SessionId": session_id})
        return results


def run(query: str, session_id: str, logger) -> str:
    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a travel itinerary specialist. Use search_web to check current "
                "attractions and practical details, then make a simple, realistic day-by-day "
                "plan that follows the user's constraints. Cite useful source URLs."
            ),
        },
        {"role": "user", "content": query},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for attractions and practical travel details.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    with logger.start_span(name="itinerary_planner_agent", set_current=False) as span:
        span.log(input=query, metadata={"SessionId": session_id})
        while True:
            with span.start_span(
                name="openai_chat_completion", type="llm", set_current=False
            ) as llm_span:
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages, tools=tools
                )
                message = response.choices[0].message
                llm_span.log(
                    input=messages,
                    output=message.model_dump(exclude_none=True),
                    metadata={
                        "model": response.model,
                        "usage": response.usage.model_dump() if response.usage else None,
                        "SessionId": session_id,
                    },
                )
            messages.append(message)
            if not message.tool_calls:
                result = message.content or "I could not make an itinerary."
                span.log(output=result, metadata={"SessionId": session_id})
                return result
            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result = search_web(arguments["query"], span, session_id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )


