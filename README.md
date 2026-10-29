# Simple multi-agent travel demo

A small terminal demo with one supervisor and two specialist agents:

- `destination_research` answers questions about a destination, attractions, food, and local recommendations.
- `itinerary_planner` makes a day-by-day trip plan.

The supervisor chooses a specialist based on the user's request. Each specialist can call its
`search_web` tool, which sends a query to Tavily and returns web results for the specialist to use.

## Run

Create a `.env` file in the repository root and add `OPENAI_API_KEY`, `TAVILY_API_KEY`, and the
three project-scoped Braintrust API keys:

```dotenv
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
BRAINTRUST_PROJECT_A_API_KEY=your-project-a-key
BRAINTRUST_PROJECT_B_API_KEY=your-project-b-key
BRAINTRUST_PROJECT_C_API_KEY=your-project-c-key
```

Project A is for supervisor delegation spans, Project B is for destination research traces,
and Project C is for itinerary planner traces. The project names default to `Project A`,
`Project B`, and `Project C`; if your Braintrust projects have different names, set
`BRAINTRUST_PROJECT_A_NAME`, `BRAINTRUST_PROJECT_B_NAME`, and `BRAINTRUST_PROJECT_C_NAME`
to match. The `.env` file is excluded by `.gitignore`. The supervisor session is one trace in
Project A, rooted at `Travel Agent`; each delegation is a child span that records the
selected agent. Specialist traces remain independent in Projects B and C. All three loggers are
initialized in the main process with separate `BraintrustState` instances and their own
project-scoped keys. This uses a lower-level Braintrust SDK API; the SDK's global helpers and
`wrap_openai` are not used for these traces. A generated `SessionId` metadata field is attached
to the supervisor, specialist, LLM, and Tavily spans so traces from one interactive session can
be correlated across all three projects. The supervisor's OpenAI calls are logged as `llm` spans;
they record the selected tool name or a response marker without logging specialist results.

Then, from the repository root:

```sh
uv run --env-file .env samsara-multi-agent
```

Example requests:

- `What are three good things to see in [city]?`
- `Plan a two-day trip focused on temples and food.`

Type `quit` to exit. Set `OPENAI_MODEL` to use a different OpenAI model; the default is
`gpt-4o-mini`.

## Layout

- `src/samsara_multi_agent/main.py` contains the supervisor and the simple tool-call loop.
- `src/samsara_multi_agent/destination_research.py` contains the destination research agent and its Tavily tool.
- `src/samsara_multi_agent/itinerary.py` contains the itinerary agent and its Tavily tool.

The specialist files are intentionally self-contained for the demo. Each Tavily search result
is returned to the calling specialist, and each specialist's final answer is returned to the
supervisor. Project A records the user's request and the selected specialist name only. Projects
B and C record their agent run, wrapped OpenAI calls, and Tavily search spans independently.
