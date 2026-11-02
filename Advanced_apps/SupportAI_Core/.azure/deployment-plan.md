# Deployment Plan: SupportAI Core → Azure Container Apps

## Project Overview
- **App:** SupportAI Core — 10-node LangGraph multi-agent support system
- **Stack:** Python 3.10, Streamlit, LangGraph, Pinecone, DeepSeek V4 Flash
- **Target:** Azure Container Apps (consumption plan, scale-to-zero)
- **Cost Target:** $0 within free tier limits
- **Mode:** MODIFY (existing project)

## Directory Strategy
- **One GitHub repo** — all files (tests, archive, evals, docs stay on disk)
- **Docker image** — only runtime files via `.dockerignore` gatekeeping
- **Zero duplication** — no separate deploy branch needed

## Files to Include in Docker Image
| File | Purpose |
|------|---------|
| `main.py` | LangGraph state machine (CPU mode, with Flashrank fix) |
| `streamlit_app.py` | Streamlit UI |
| `logger.py` | SQLite + JSONL logging |
| `kb.json` | Knowledge base (1,137 QA pairs) |
| `requirements.txt` | Python dependencies (pinned versions) |
| `Dockerfile` | Container build (non-root user, health check) |
| `.dockerignore` | Build context filter |
| `.streamlit/config.toml` | Streamlit configuration |

## Files Excluded from Docker (via .dockerignore)
`tests/`, `archive/`, `data/`, `Evals/`, `docs/`, `config/`, `README.md`, `.gitignore`, `.env`, `__pycache__/`, `*.pyc`

## Modifications Completed
| File | Change | Status |
|------|--------|--------|
| `.dockerignore` | Add dev-only dirs to exclusion list | ✅ Done |
| `.streamlit/config.toml` | Created from original theme + server options | ✅ Done |
| `Dockerfile` | Removed build-essential, added non-root user, health check | ✅ Done |
| `main.py` | Added `FlashrankRerank.model_rebuild()` for Pydantic v2 compat | ✅ Done |
| `requirements.txt` | Pinned exact working versions, fixed pinecone version conflict | ✅ Done |

## Architecture

### Components Map
```
GitHub Repo → Docker build → Azure Container Registry → Container Apps
                                                              ↓
                                                    Streamlit on :8501
                                                              ↓
                                              ┌─────────────────┐
                                              │ Pinecone (API)  │
                                              │ DeepSeek (API)  │
                                              └─────────────────┘
```

### Azure Services
| Service | Purpose | Cost |
|---------|---------|------|
| **Azure Container Apps** | Streamlit app hosting (consumption plan) | Free tier: 180K vCPU-s + 360K GiB-s + 2M req/mo |
| **Azure Container Registry** | Docker image storage (Basic SKU) | Free: 100 GB storage for 12 months |
| **No database needed** | MemorySaver (in-process) + SQLite (ephemeral) | $0 |

### External APIs (not Azure)
| Service | Purpose | Free Tier |
|---------|---------|-----------|
| **Pinecone Serverless** | Vector database for RAG | 1 free index, limited vectors |
| **OpenCode Zen** | DeepSeek V4 Flash LLM | Free tier (no billing setup needed) |

## Scaling Configuration
- **Plan:** Consumption (serverless) — scale-to-zero when idle
- **Min replicas:** 0 (no cost when unused)
- **Max replicas:** 1 (prevents accidental scaling)
- **CPU:** 1 vCPU (4 GB RAM)
- **Cold start:** ~15-25s (models are pre-cached in Docker layers)

**Trade-off accepted:** MemorySaver state is lost on scale-to-zero. Each visitor starts a fresh session. This is acceptable for a CV demo.

## Env Vars Required
| Variable | Source | Purpose |
|----------|--------|---------|
| `PINECONE_API_KEY` | Pinecone console | Vector DB access |
| `ZEN_API_KEY` | OpenCode Zen | DeepSeek LLM access |
| `LANGCHAIN_TRACING_V2` | — | Set to `false` |

## Deployment Instructions
1. Build Docker image on user's machine: `docker build -t supportai-core .`
2. Tag & push to ACR
3. Deploy with: `az containerapp create ...`

## Azure Context
- **Subscription:** To be set during `azd init`
- **Location:** southeastasia (Singapore) — best latency for India

## Infrastructure Files Generated
| File | Purpose |
|------|---------|
| `azure.yaml` | AZD service configuration |
| `infra/main.bicep` | Root Bicep template — orchestrates all resources |
| `infra/modules/log-analytics.bicep` | Log Analytics Workspace |
| `infra/modules/container-apps-environment.bicep` | Container Apps Environment |
| `infra/modules/container-registry.bicep` | Azure Container Registry |
| `infra/modules/container-app.bicep` | Container App (minReplicas: 0, maxReplicas: 1) |
| `infra/modules/acr-pull-role.bicep` | ACR Pull role assignment |

## Cost Analysis
| Period | Cost | Details |
|--------|------|---------|
| First 30 days | $0 | $200 free credit covers any overages |
| Months 1-12 | $0 | Container Apps always-free + ACR Basic free |
| After 12 months | ~$0.40/mo | ACR storage for 4 GB image |
| If app goes viral | ~$5-15/mo | Additional Container Apps compute |

## Deployment Script (summary)
```bash
az login
azd init
azd env set AZURE_LOCATION southeastasia
azd up
az containerapp registry set --name <app-name> --resource-group <rg-name> --server <acr-server> --identity system
azd deploy
```

**Status:** `Ready for Validation`
