"""A tiny supervisor loop that routes a user request to specialist agents."""

import json
import os
import uuid
from functools import partial

from braintrust import BraintrustState, init_logger
from openai import OpenAI

from . import destination_research, itinerary


def _supervisor_logger():
    api_key = os.getenv("BRAINTRUST_PROJECT_A_API_KEY")
    if not api_key:
        raise RuntimeError("Set BRAINTRUST_PROJECT_A_API_KEY in your environment.")
    return init_logger(
        project=os.getenv("BRAINTRUST_PROJECT_A_NAME", "Project A"),
        api_key=api_key,
        # Each project-scoped key needs its own SDK login state in this process.
        state=BraintrustState(),
        set_current=False,
    )


def _run_session(client, supervisor_logger, sub_agents, session_id) -> None:
    session_requests = []
    delegation_actions = []

    with supervisor_logger.start_span(
        name="Travel Agent", type="task", set_current=False
    ) as session_span:
        session_span.log(metadata={"SessionId": session_id})
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in {"quit", "exit"}:
                break
            if not query:
                continue
            session_requests.append(query)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a supervisor. Route each user request to exactly one specialist: "
                        "destination_research for questions about a place, attractions, food, or "
                        "local recommendations; itinerary_planner when the user asks to plan or "
                        "organize a trip. After the specialist returns, give its answer to the "
                        "user. Do not answer from your own knowledge."
                    ),
                },
                {"role": "user", "content": query},
            ]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "destination_research",
                        "description": "Research a destination, attractions, food, or local recommendations.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "itinerary_planner",
                        "description": "Plan or organize a trip into a day-by-day itinerary.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                },
            ]

            while True:
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages, tools=tools
                )
                message = response.choices[0].message
                messages.append(message)
                if not message.tool_calls:
                    print(f"\nAssistant: {message.content or ''}")
                    break

                for tool_call in message.tool_calls:
                    specialist = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    if specialist in sub_agents:
                        label = (
                            "destination research agent"
                            if specialist == "destination_research"
                            else "itinerary planner agent"
                        )
                        print(f"\n[Supervisor] Delegating to {label}...")
                        with session_span.start_span(
                            name="supervisor_delegation", type="task", set_current=False
                        ) as span:
                            span.log(
                                input=query,
                                output={"delegated_to": specialist},
                                metadata={"SessionId": session_id},
                            )
                        delegation_actions.append({"request": query, "delegated_to": specialist})
                        result = sub_agents[specialist](arguments["query"], session_id)
                    else:
                        result = "Unknown specialist."
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    )

        session_span.log(
            input={"requests": session_requests},
            output={"delegations": delegation_actions},
            metadata={"SessionId": session_id},
        )


def main() -> None:
    client = OpenAI()
    supervisor_logger = _supervisor_logger()
    destination_logger = destination_research.create_logger()
    itinerary_logger = itinerary.create_logger()
    sub_agents = {
        "destination_research": partial(destination_research.run, logger=destination_logger),
        "itinerary_planner": partial(itinerary.run, logger=itinerary_logger),
    }
    print("Travel demo. Ask about a destination or request an itinerary. Type 'quit' to exit.")
    session_id = str(uuid.uuid4())
    _run_session(client, supervisor_logger, sub_agents, session_id)


if __name__ == "__main__":
    main()
