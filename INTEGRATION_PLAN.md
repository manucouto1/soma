# Soma Integration Plan

## Vision

Soma becomes the **research engine** behind Chatty the Lab. Together they form a platform where autonomous agents can design, execute, analyze, and document scientific experiments — with full visual supervision, prompt optimization, and a navigable knowledge base.

## Architecture

```
┌─────────────── CHATTY THE LAB ──────────────────────┐
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           Agent Studio (existing)             │   │
│  │                                               │   │
│  │  [Agent] ──► [SomaResearch] ──► [Agent]      │   │
│  │                    │                          │   │
│  │         Uses soma-mcp tools                   │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐   │
│  │        New UI Panels                          │   │
│  │                                               │   │
│  │  ┌─────────────┐  ┌──────────────────────┐   │   │
│  │  │  Pipeline    │  │  Experiment          │   │   │
│  │  │  Designer    │  │  Dashboard (W&B)     │   │   │
│  │  └─────────────┘  └──────────────────────┘   │   │
│  │  ┌─────────────┐  ┌──────────────────────┐   │   │
│  │  │  Knowledge   │  │  Graph Version       │   │   │
│  │  │  Explorer    │  │  Control             │   │   │
│  │  └─────────────┘  └──────────────────────┘   │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │     Graph Versioning System (new)             │   │
│  │                                               │   │
│  │  Tracks: topology, nodes, configs, prompts    │   │
│  │  Links: version → experiments → metrics       │   │
│  │  Enables: prompt optimization against metrics │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
└──────────────────────┬───────────────────────────────┘
                       │ MCP Protocol
                       │
┌──────────────────────▼───────────────────────────────┐
│                 SOMA MCP SERVER                       │
│                                                       │
│  Tools:                    Resources:                 │
│  ├── Code                  ├── experiment://           │
│  │   read_filter_source    │   list, {id}/metrics     │
│  │   write_filter_source   │   {id}/code_diff         │
│  │   list_filters          ├── research_line://        │
│  │   get_filter_schema     │   {name}/trajectory      │
│  ├── Execution             ├── dashboard://            │
│  │   run_pipeline          │   study/{id}              │
│  │   run_study             │   parallel_coords/{id}    │
│  │   get_trial_metrics     ├── report://               │
│  ├── Knowledge             │   {line}/latest           │
│  │   record_experiment     │   {line}/full             │
│  │   query_kb              └── graph_version://        │
│  │   get_trajectory            {id}/config             │
│  │   document_finding          {id}/diff               │
│  ├── Project                                          │
│  │   git_commit_experiment                            │
│  │   create_research_line                             │
│  │   generate_report                                  │
│  └── Prompt Optimization                              │
│      evaluate_prompt                                  │
│      run_prompt_study                                 │
│      compare_prompt_versions                          │
│                                                       │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                 SOMA ENGINE                           │
│  soma-core, soma-compiler, soma-runtime,             │
│  soma-memory (ChronosVector), soma-worker            │
└──────────────────────────────────────────────────────┘
```

## Phases

### Phase 1: soma-mcp (in Soma project)

**Goal**: Agent can run experiments, query knowledge, modify code via MCP tools.

| Step | Description | Effort |
|------|-------------|--------|
| 1.1 | MCP server scaffold (stdio transport) | Low |
| 1.2 | Code tools: read/write/list filters in a project directory | Medium |
| 1.3 | Execution tools: run_pipeline, run_study, get_results | Medium |
| 1.4 | Knowledge tools: record, query, trajectory, change_points | Medium |
| 1.5 | Project tools: git commit per experiment, research line management | Medium |
| 1.6 | Report generation: markdown reports from knowledge base | Medium |
| 1.7 | Prompt optimization tools: evaluate_prompt, run_prompt_study | High |

**Deliverable**: An agent connected to soma-mcp can autonomously research.

### Phase 2: Graph Versioning (in Chatty project)

**Goal**: Every graph configuration is versioned. Prompt changes tracked. Versions linked to experiment metrics.

| Step | Description | Effort |
|------|-------------|--------|
| 2.1 | `graph_versions` table: stores topology + node configs + prompts as snapshots | Medium |
| 2.2 | Auto-version on graph save: diff detection, semantic versioning | Medium |
| 2.3 | Version diff view: what changed between v1 and v2 (nodes, edges, prompts) | High |
| 2.4 | Link versions to experiments: "this graph version produced these metrics" | Medium |
| 2.5 | Prompt versioning: track prompt text changes per node independently | Medium |
| 2.6 | Prompt A/B testing: run same graph with different prompts, compare metrics | High |
| 2.7 | API endpoints: CRUD for graph versions, diffs, experiment links | Medium |

**Deliverable**: Full audit trail of graph evolution linked to experimental results.

### Phase 3: Prompt Optimization (spans both projects)

**Goal**: Optimize prompts against metrics like any other hyperparameter.

| Step | Description | Where | Effort |
|------|-------------|-------|--------|
| 3.1 | PromptFilter in Soma: a filter that wraps an LLM call with a prompt template | Soma | Medium |
| 3.2 | PromptSearchSpace: define prompt variations (template slots, phrasing alternatives) | Soma | High |
| 3.3 | Prompt Study: optimize prompt parameters via Bayesian/grid search against eval metrics | Soma | High |
| 3.4 | Eval node integration: use Chatty's 19 eval metrics as optimization objectives | Chatty | Medium |
| 3.5 | Prompt diff visualization: highlight what changed between prompt versions | Chatty | Medium |
| 3.6 | Prompt performance dashboard: metric evolution across prompt versions | Chatty | High |

**Deliverable**: Researcher says "optimize this prompt for F1 score" → system explores variations → finds best.

### Phase 4: Chatty UI Integration

**Goal**: Visual supervision and exploration of research.

| Step | Description | Effort |
|------|-------------|--------|
| 4.1 | SomaResearch node type: configures and launches research from Agent Studio | High |
| 4.2 | Experiment Dashboard panel: real-time metrics, learning curves, parallel coords | High |
| 4.3 | Knowledge Explorer panel: browse research lines, experiment trees, generated docs | High |
| 4.4 | Pipeline Designer panel: visual filter composition with schema validation | High |
| 4.5 | Graph Version panel: timeline of graph changes, diff view, linked experiments | High |
| 4.6 | Code diff panel: see filter source changes per experiment | Medium |

**Deliverable**: Full visual research IDE.

### Phase 5: Agent Skills (in Chatty project)

**Goal**: Agent knows HOW to do research, not just WHAT tools to call.

| Step | Description | Effort |
|------|-------------|--------|
| 5.1 | research_methodology skill: experimental design, hypothesis formulation | Low |
| 5.2 | code_modification skill: safe code changes, testing, rollback | Low |
| 5.3 | documentation skill: scientific writing, structured findings | Low |
| 5.4 | prompt_engineering skill: prompt optimization strategies, evaluation | Low |
| 5.5 | analysis skill: interpret metrics, detect patterns, recommend next steps | Low |

**Deliverable**: Agent produces research-quality work without hand-holding.

## Prompt Optimization Detail

This is the most novel aspect. The idea:

```
A prompt in a Chatty graph node is a HYPERPARAMETER.

Just like learning_rate in an SVM, the prompt text can be:
- Versioned (tracked per change)
- Searched (variations explored)
- Optimized (against eval metrics)
- Documented (which version worked best and why)

The same Bayesian optimization that Soma uses for
hyperparameters works for prompt parameters.
```

### How it works

```
1. User defines a graph with LLM nodes containing prompt templates:
   "Summarize the following text in {style}: {input}"

2. Prompt parameters become search dimensions:
   style: Categorical["concise", "detailed", "academic", "bullet_points"]

   Or template variations:
   prompt_v1: "Summarize: {input}"
   prompt_v2: "Write a {style} summary of: {input}"
   prompt_v3: "As an expert, create a {style} summary: {input}"

3. Soma runs a Study:
   - Each trial = one prompt version
   - Eval metrics = ROUGE, F1, human_rating, etc.
   - Bayesian sampler explores prompt space

4. Results tracked in knowledge base:
   - "prompt_v2 with style=academic scored 0.92 ROUGE-L"
   - "prompt_v3 underperforms on short texts"

5. Graph version updated with best prompt
```

### Graph Version Schema

```sql
CREATE TABLE graph_versions (
    id UUID PRIMARY KEY,
    graph_id UUID REFERENCES agent_graphs(id),
    version_number INT NOT NULL,
    -- Full snapshot
    topology JSONB NOT NULL,        -- nodes + edges (structure only)
    node_configs JSONB NOT NULL,    -- all node configurations
    prompts JSONB NOT NULL,         -- extracted prompts per node
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,                -- user or agent
    message TEXT,                   -- "Optimized summarizer prompt"
    parent_version UUID,            -- for branching
    -- Experiment link
    experiment_id TEXT,             -- soma experiment ID
    metrics JSONB,                  -- snapshot of metrics at this version

    UNIQUE(graph_id, version_number)
);

CREATE TABLE prompt_versions (
    id UUID PRIMARY KEY,
    graph_version_id UUID REFERENCES graph_versions(id),
    node_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,       -- for dedup
    -- Optimization
    search_space JSONB,             -- template variables and their ranges
    best_params JSONB,              -- optimized values
    eval_metrics JSONB,             -- metrics for this specific prompt

    UNIQUE(graph_version_id, node_id)
);
```

## Data Flow Example

```
Researcher: "Optimize the summarizer in my RAG pipeline"

Agent (using soma-mcp tools):

1. read_filter_source("rag_pipeline/summarizer.py")
   → Reads current implementation

2. get_filter_schema("summarizer")
   → Understands input/output types

3. create_research_line("summarizer_optimization")
   → Creates tracking context

4. run_prompt_study({
     graph_id: "rag_pipeline",
     node_id: "summarizer",
     prompt_template: "Summarize: {input}",
     search_space: {
       style: ["concise", "detailed", "academic"],
       instruction: ["Summarize", "Extract key points from", "Write a brief overview of"]
     },
     eval_metrics: ["rouge_l", "bleu", "human_coherence"],
     n_trials: 30
   })
   → Runs optimization study

5. record_experiment({
     name: "summarizer_opt_v1",
     line: "summarizer_optimization",
     metrics: { rouge_l: 0.87, bleu: 0.72 },
     best_prompt: "Extract key points from {input} in an academic style"
   })
   → Records result

6. document_finding({
     line: "summarizer_optimization",
     finding: "Academic style with 'Extract key points' instruction improved ROUGE-L by 15% over baseline"
   })
   → Documents insight

7. generate_report("summarizer_optimization")
   → Creates navigable markdown report

8. git_commit_experiment("summarizer_opt_v1")
   → Commits the winning prompt to version control
```

## Priority Order

```
Phase 1 (soma-mcp)         ← START HERE. Unblocks everything.
Phase 2 (graph versioning)  ← In parallel, in Chatty.
Phase 3 (prompt optim)      ← Connects both. Most novel.
Phase 4 (UI panels)         ← Incremental, per panel.
Phase 5 (agent skills)      ← Quick wins, low effort.
```
