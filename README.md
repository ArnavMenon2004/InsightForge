# InsightForge

> A production-quality multi-agent AI research assistant demonstrating clean architecture, LangGraph orchestration, and SOLID engineering principles.

---

## What it does

InsightForge accepts a research topic and runs it through a five-agent LangGraph pipeline, producing a fully structured, fact-checked, and cited research report — exported as a PDF.

This is **not** a chatbot wrapper. It is a modular agent system built to demonstrate how production AI systems are architected.

---

## Pipeline

```
User Query
    │
    ▼
┌─────────────┐
│   Planner   │  Produces: ResearchPlan
│             │  (topic, audience, depth, sections, keywords)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│   Researcher     │  Per section:
│                  │   1. LLM → focused search queries
│                  │   2. WebSearchTool → Source objects
│                  │   3. WebFetchTool  → full page content
│                  │   4. LLM → section summary
│                  │  Produces: ResearchDocument
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│   Writer    │  Section-by-section LLM calls
│             │  Produces: FinalReport (draft)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Fact Checker    │  Verifies claims against sources
│                  │  Assigns SUPPORTED / UNSUPPORTED / UNVERIFIABLE
│                  │  Produces: FactCheckReport
└──────┬───────────┘
       │
       ▼
┌────────────────┐
│   Citation     │  Maps claims to source indices
│                │  Produces: CitedClaim list
└──────┬─────────┘
       │
       ▼
  FinalReport → PDF Export
```

All agents communicate exclusively via **typed Pydantic models**. No raw dictionaries cross agent boundaries.

---

## Folder structure

```
InsightForge/
├── agents/
│   ├── base_agent.py       # Abstract base — timing, logging, AgentResult
│   ├── planner.py          # Single-chain agent
│   ├── researcher.py       # Multi-step orchestration loop
│   ├── writer.py           # Section-by-section writing
│   ├── fact_checker.py     # Claim verification
│   └── citation.py        # Source mapping
├── graphs/
│   └── pipeline.py         # LangGraph StateGraph wiring all agents
├── prompts/
│   ├── planner.py
│   ├── researcher.py       # Query generation + summarisation prompts
│   ├── writer.py           # Section writer + metadata prompts
│   ├── fact_checker.py
│   └── citation.py
├── schemas/
│   ├── state.py            # GraphState — LangGraph shared state (TypedDict)
│   ├── research.py         # ResearchPlan, ResearchDocument, SearchQueries, etc.
│   ├── report.py           # FinalReport, FactCheckReport, CitedClaim
│   ├── result.py           # AgentResult (generic wrapper)
│   └── source.py           # Source, SourceType
├── providers/
│   └── tavily_provider.py  # Tavily API — isolated from business logic
├── tools/
│   ├── base_tool.py        # Abstract tool interface
│   ├── web_search.py       # WebSearchTool — wraps TavilyProvider
│   ├── web_fetch.py        # WebFetchTool — full page content extraction
│   └── export.py           # PDFExporter
├── config/
│   ├── settings.py         # Environment variables + token budget constants
│   └── llm.py              # Gemini LLM initialisation
├── utils/
│   └── logger.py           # Rotating file + console logger
├── tests/
│   ├── test_researcher.py
│   ├── test_writer.py
│   ├── test_fact_checker.py
│   ├── test_citation.py
│   └── test_web_search.py
├── app.py                  # Streamlit UI
├── .env.example
└── requirements.txt
```

---

## Design principles

### Layered architecture

```
Provider → Tool → Agent → Graph → UI
```

Each layer knows only about the layer to its left. The UI never touches a provider. Agents never touch Tavily directly.

### Multi-provider LLM routing

InsightForge runs every LLM call on Google Gemini:

| Provider | Model | Used for |
|---|---|---|
| Google Gemini | gemini-3.1-flash-lite | Planner, Researcher, Writer, FactChecker, Citation |

Gemini's free tier provides 250K TPM and 500 RPD — sufficient for ~38 full pipeline runs per day with no rate-limit errors.

### Dependency inversion

Every tool and agent accepts dependencies via constructor injection with sensible defaults. This makes every class testable without API keys.

```python
# In tests — pass a mock
agent = ResearcherAgent(search_tool=mock_tool, fetch_tool=mock_fetch)

# In production — defaults used automatically
agent = ResearcherAgent()
```

### Typed contracts everywhere

Every piece of data exchanged between agents is a Pydantic model. Raw dictionaries do not cross module boundaries.

### Section-by-section writing

The WriterAgent makes one focused LLM call per section rather than sending all summaries in one massive prompt. This eliminates single-request token ceiling errors and improves output quality by giving each section the model's full attention.

### Source quality filtering

WebSearchTool discards Tavily results below a configurable relevance score threshold (`MIN_SOURCE_SCORE` in `config/settings.py`), preventing SEO spam from reaching the summarisation step.

### Full-page content fetching

WebFetchTool fetches the actual article body (up to `SNIPPET_CHAR_LIMIT` characters) rather than relying on Tavily's short snippets. This gives the summarisation LLM substantially more material to work with, producing richer section content.

---

## Stack

| Layer | Technology |
|---|---|
| LLM (primary) | Google Gemini — gemini-3.1-flash-lite |
| Orchestration | LangGraph |
| LLM framework | LangChain |
| Web search | Tavily |
| Page fetching | httpx + BeautifulSoup4 |
| Data validation | Pydantic v2 |
| UI | Streamlit |
| Export | PDF (reportlab) |
| Logging | Python logging with rotating file handler |

---

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/InsightForge.git
cd InsightForge

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Edit .env and add your keys

# 5. Run
streamlit run app.py
```

**.env**
```
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## API keys

| Service | Free tier | Link |
|---|---|---|
| Google Gemini | 250K TPM · 500 RPD · no billing required | https://aistudio.google.com |
| Tavily | 1,000 searches/month | https://tavily.com |

---

## Running tests

```bash
pytest tests/ -v
```

All tests mock external API calls (Gemini + Tavily) so they run offline without API keys.

---

## Token budget

A full pipeline run on the free tier uses approximately:

| Agent | LLM calls | ~Tokens |
|---|---|---|
| Planner | 1 | ~650 |
| Researcher | 10 (2 per section) | ~10,200 |
| Writer | 6 (1 per section + metadata) | ~7,300 |
| Fact Checker | 1 | ~3,500 |
| Citation | 1 | ~2,400 |
| **Total** | **19** | **~24,000** |

Gemini's free tier (250K TPM, 500 RPD) comfortably supports ~38 full runs per day.