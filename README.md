# Black Sheep Ventures — Collective Intelligence & Multi-Agent Prediction Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPLv3-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Engine: OASIS](https://img.shields.io/badge/Engine-CAMEL--OASIS-orange.svg)](https://github.com/camel-ai/oasis)
[![Memory: Zep Cloud](https://img.shields.io/badge/Graph-Zep%20Cloud-purple.svg)](https://app.getzep.com/)

An autonomous multi-agent simulation and market prediction engine engineered by **Black Sheep Venture Labs**. The system ingests seed documents, builds temporal knowledge graphs, spins up synthetic agent populations across simulated social networks (Twitter and Reddit), and deploys autonomous ReACT intelligence agents to synthesize rigorous market predictions and strategic risk assessments.

---

## 🏛️ System Architecture

```
                                  [ Input Documents / Thesis ]
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  Ontology & Graph     │
                                    │  Ingestion Service    │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  Synthetic Population │
                                    │  Generator (Demogr.)  │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │          Dual-Platform Social Simulation            │
                     │  ┌───────────────────────┐ ┌──────────────────────┐ │
                     │  │ Twitter Stream Driver │ │ Reddit Stream Driver │ │
                     │  └───────────┬───────────┘ └──────────┬───────────┘ │
                     └──────────────┼────────────────────────┼─────────────┘
                                    │                        │
                                    ▼                        ▼
                                    ┌────────────────────────┐
                                    │ Zep Cloud Temporal     │
                                    │ Knowledge Graph Memory │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │ Autonomous ReACT       │
                                    │ Intelligence Agent     │
                                    │ (InsightForge, Search, │
                                    │ Panorama, Interviews)  │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    [ Strategic Prediction Report ]
```

---

## ⚡ Key Capabilities

1. **Document Intake & Domain Ontology Mining**: Automatically extracts domain-specific entities (e.g. Founders, Venture Capitalists, Regulators, Tech Directors) and relationship matrices from raw text (PDF, Markdown, TXT).
2. **Synthetic Population Generation**: Generates rich persona profiles with demographic distribution, psychological MBTI traits, career profiles, and initial sentiments.
3. **Dual-Channel Social Dynamics**: Runs parallelized CAMEL-AI OASIS simulation loops across Twitter (fast-paced feeds, reposts, quotes) and Reddit (threaded discussions, upvotes/downvotes, long-form critiques).
4. **Temporal Knowledge Graph Updating**: Live-syncs agent interactions, ideological shifts, and consensus milestones into Zep Cloud temporal memory.
5. **Autonomous ReACT Intelligence Synthesis**: Employs dynamic reasoning loops with multi-modal tools:
   - `insight_forge`: Sub-query breakdown and causal correlation.
   - `panorama_search`: Macro-level graph statistics and entity network shifts.
   - `quick_search`: Fast semantic and factual fact verification.
   - `interview_agents`: Direct interrogations and 1-on-1 qualitative interviews with simulated agents.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11 or 3.12
- `uv` (recommended) or `pip`
- Zep Cloud API Key ([Get Zep Key](https://app.getzep.com/))
- LLM API Key (Google Gemini via OpenAI-compat endpoint, OpenAI, or DeepSeek)

### Installation

```bash
# Clone the repository
git clone https://github.com/BlackSheepVentures/blacksheep-engine.git
cd blacksheep-engine

# Copy configuration
cp .env.example .env

# Install dependencies using uv
uv sync
```

### Configure `.env`

```ini
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL_NAME=gemini-3.5-flash-lite
ZEP_API_KEY=your_zep_api_key_here
FLASK_PORT=5001
```

### Running the Engine

```bash
uv run python run.py
```

The server initializes on `http://0.0.0.0:5001` with complete OpenAPI-compliant endpoints.

---

## 📡 REST API Reference

### 1. Knowledge Graph & Project Management (`/api/graph`)
- `POST /api/graph/project/<project_id>/reset`: Reset project state.
- `POST /api/graph/ontology/generate`: Generate 10-type domain ontology from uploaded text.
- `POST /api/graph/build`: Chunk text and construct temporal knowledge graph.
- `GET /api/graph/task/<task_id>`: Check asynchronous graph build task status.
- `GET /api/graph/data/<graph_id>`: Retrieve nodes, edges, and statistics for graph visualization.

### 2. Multi-Agent Simulation (`/api/simulation`)
- `POST /api/simulation/create`: Initialize new simulation instance.
- `POST /api/simulation/prepare`: Validate entities and compile simulation scripts.
- `POST /api/simulation/generate-profiles`: Generate realistic agent personas.
- `POST /api/simulation/start`: Launch parallelized OASIS simulation.
- `POST /api/simulation/stop`: Gracefully terminate running simulation.
- `GET /api/simulation/<id>/run-status`: Poll simulation execution metrics, rounds, and active agents.
- `GET /api/simulation/<id>/timeline`: Retrieve round-by-round event timeline.
- `GET /api/simulation/<id>/posts`: Inspect simulated social media posts.
- `POST /api/simulation/interview`: Conduct interactive 1-on-1 interview with any simulated agent.

### 3. Intelligence Report Generation (`/api/report`)
- `POST /api/report/generate`: Launch autonomous ReACT report synthesis agent.
- `GET /api/report/<report_id>`: Retrieve complete structured Markdown report.
- `GET /api/report/<report_id>/progress`: Real-time section and tool invocation progress.
- `GET /api/report/<report_id>/agent-log/stream`: Server-Sent Events (SSE) live reasoning stream.
- `POST /api/report/chat`: Interactive conversational Q&A over the generated intelligence report.

---

## ⚖️ Legal & Intellectual Property Notice

This software is developed and maintained by **Black Sheep Venture Labs**. 
It incorporates open-source collective intelligence principles and is licensed under the **GNU Affero General Public License Version 3 (AGPL-3.0)**. 

Derivative works, enhancements, and deployments over computer networks must comply with AGPL-3.0 copyleft obligations and preserve all original copyright notices.
