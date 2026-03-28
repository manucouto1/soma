# The Nous-Soma-Chronos Ecosystem

## Philosophy

Three Greek concepts, three projects, three responsibilities:

```
Nous  (νοῦς)   = Mind     → Understands, reasons, decides, plans
Soma  (σῶμα)   = Body     → Executes, materializes, caches, optimizes
Chronos (χρόνος) = Time/Memory → Remembers, evolves, analyzes temporally
```

## Architecture

```
┌──────────── NOUS ─────────────────────────┐
│  The Mind: understands, reasons, decides   │
│                                            │
│  ├── Agent Studio                          │
│  │   Visual graph editor for reasoning     │
│  │                                         │
│  ├── Agent Runtime                         │
│  │   OpenFang: skills, hands, memory       │
│  │                                         │
│  ├── Evaluation Lab                        │
│  │   Model comparison, A/B testing         │
│  │   Reward modeling, human feedback       │
│  │                                         │
│  ├── Knowledge Base                        │
│  │   Research lines, trajectory analysis   │
│  │   Change points, insights, reports      │
│  │   Promising line detection              │
│  │   Prompt versioning & optimization      │
│  │                                         │
│  ├── Graph Versioning                      │
│  │   Topology, configs, prompts tracked    │
│  │   Linked to experiments and metrics     │
│  │                                         │
│  └── Frontend (SvelteKit)                  │
│      Dashboard, experiment panels, UI      │
│                                            │
└────────────┬──────────────────────────────┘
             │ MCP Protocol
             │
┌────────────▼──────────────────────────────┐
│          SOMA                              │
│  The Body: executes and materializes       │
│                                            │
│  ├── Filters & Pipelines                   │
│  │   fit/forward lifecycle, composition    │
│  │                                         │
│  ├── Compiler                              │
│  │   Graph → ExecutionPlan                 │
│  │   Cache resolution, schema validation   │
│  │                                         │
│  ├── Runtime                               │
│  │   Parallel executor, stream processing  │
│  │   Event bus, metrics reporting          │
│  │                                         │
│  ├── Optimization                          │
│  │   Grid, Random, Bayesian samplers       │
│  │   Median, Percentile pruners            │
│  │                                         │
│  ├── Caching & Virtualization              │
│  │   LRU memory, local disk, S3            │
│  │   VirtualValue (lazy materialization)   │
│  │                                         │
│  ├── Experiment Store                      │
│  │   Record what was executed and results  │
│  │   (NOT knowledge — just data)           │
│  │                                         │
│  ├── Workers                               │
│  │   Distributed execution, Axum server    │
│  │                                         │
│  ├── MCP Server                            │
│  │   Tools for execution & data access     │
│  │                                         │
│  └── Python Bindings                       │
│      pip install soma                      │
│                                            │
└────────────┬──────────────────────────────┘
             │
┌────────────▼──────────────────────────────┐
│      CHRONOS VECTOR                        │
│  The Memory: remembers and evolves         │
│                                            │
│  ├── Temporal HNSW Index                   │
│  │   Semantic + temporal distance          │
│  │                                         │
│  ├── Analytics                             │
│  │   Velocity, drift, change points        │
│  │                                         │
│  ├── Tiered Storage                        │
│  │   Hot (RocksDB) → Warm → Cold (S3)     │
│  │                                         │
│  ├── MCP Server                            │
│  │   Temporal search and analysis tools    │
│  │                                         │
│  └── Python Bindings                       │
│      pip install chronos-vector            │
│                                            │
└───────────────────────────────────────────┘
```

## Responsibility Boundaries

### What lives WHERE

| Concept | Soma | Nous | ChronosVector |
|---------|------|------|---------------|
| Filter/Pipeline definition | ✓ | | |
| Pipeline execution | ✓ | | |
| Execution caching | ✓ | | |
| Hyperparameter optimization | ✓ | | |
| Stream processing | ✓ | | |
| Experiment record (raw data) | ✓ | | |
| Agent graphs (reasoning) | | ✓ | |
| Agent runtime (OpenFang) | | ✓ | |
| Knowledge organization | | ✓ | |
| Research line management | | ✓ | |
| Trajectory analysis | | ✓ | |
| Insight generation | | ✓ | |
| Report generation | | ✓ | |
| Reward modeling | | ✓ | |
| Prompt optimization | | ✓ | |
| Graph versioning | | ✓ | |
| Evaluation (A/B, human) | | ✓ | |
| Temporal vector storage | | | ✓ |
| Semantic search | | | ✓ |
| Change point detection | | | ✓ |
| Drift analysis | | | ✓ |

### Data flow

```
Nous decides what to do
  → calls Soma via MCP to execute pipelines
    → Soma records experiment data
    → Soma stores results in cache
    → Soma returns metrics to Nous

Nous analyzes results
  → queries ChronosVector for temporal patterns
  → compares with previous experiments
  → detects trends, change points
  → decides next experiment

Nous documents findings
  → stores insights in its own knowledge base
  → generates reports
  → links to graph versions
  → trains reward models from evaluations
```

## Projects

### Chatty the Lab (existing, unchanged)

Repository: `github.com/manucouto1/chatty-the-lab`

Stays as-is. Serves as the codebase foundation for Nous. Will continue
to work independently for anyone using it today.

### Nous (new, forked from Chatty conceptually)

Repository: `github.com/manucouto1/nous` (to be created)

Built from Chatty's foundations but restructured:
- Studio and Library sections preserved and extended
- Chats section evolved into Evaluation Lab
- Graph versioning system added
- Knowledge base (intelligent layer) added
- Soma and ChronosVector integration via MCP
- Reward modeling and prompt optimization

### Soma (existing)

Repository: `github.com/manucouto1/soma`

Computational graph runtime. Current scope stays but knowledge
management moves to Nous:
- `soma-memory` becomes `soma-experiments` (just experiment records)
- `soma-agent` stays as a basic agent loop (Nous has the smart agent)
- `soma-mcp` exposes execution and experiment data tools
- Knowledge tools (trajectory, reports, etc.) will be in Nous

### ChronosVector (existing)

Repository: `github.com/manucouto1/chronos-vector`

Temporal vector database. No changes needed. Both Nous and Soma
use it as a storage backend via its Rust crate and MCP server.

## Migration Plan

### What moves from Soma to Nous

When Nous is created, these concerns migrate:

| Current Soma module | Moves to | Reason |
|---------------------|----------|--------|
| `soma-memory/knowledge_base.rs` (KnowledgeBase trait) | Nous | Knowledge is cognitive |
| `soma-memory/chronos_kb.rs` | Nous | Intelligent analysis is cognitive |
| `soma-agent/` | Nous (enhanced) | Research agent is a reasoning concern |
| `soma-mcp/` knowledge tools | Nous MCP | Analysis and reports are cognitive |
| Research line management | Nous | Strategic decisions |
| Report generation | Nous | Communication |
| Trajectory analysis | Nous | Interpretation |

### What stays in Soma

| Module | Stays because |
|--------|--------------|
| `soma-core/` | Types and traits for computation |
| `soma-compiler/` | Plan compilation is execution |
| `soma-runtime/` | Execution engine |
| `soma-memory/record.rs` | ExperimentRecord is execution data |
| `soma-worker/` | Distributed execution |
| `soma-python/` | Python bindings for execution |
| `soma-mcp/` execution tools | Running pipelines and studies |
| Basic experiment store | Recording what was executed |

### What Soma's MCP exposes (post-migration)

```
Execution tools:
  run_pipeline          → execute a pipeline
  run_study             → run optimization study
  get_study_results     → retrieve study metrics

Experiment store tools:
  record_experiment     → store execution results
  get_experiment        → retrieve by ID
  list_experiments      → list all (flat, no analysis)

Code tools:
  list_filters          → find filter files
  read_filter_source    → read source code
  write_filter_source   → modify source code
```

### What Nous's MCP exposes (new)

```
Knowledge tools:
  create_research_line  → organize experiments
  get_trajectory        → metric evolution
  get_change_points     → detect breakthroughs
  promising_lines       → what's worth continuing
  query_knowledge       → semantic search over knowledge
  document_finding      → record an insight

Evaluation tools:
  evaluate_response     → score a response
  compare_versions      → A/B test graph versions
  train_reward_model    → learn from human feedback
  predict_reward        → score with learned model

Report tools:
  generate_report       → markdown from research line
  generate_comparison   → side-by-side version comparison

Graph versioning tools:
  save_graph_version    → snapshot current graph
  diff_versions         → what changed
  link_to_experiment    → connect version to results
  list_versions         → version history
```

## Implementation Order

```
1. [Soma]    Rename soma-memory → soma-experiments (just ExperimentStore)
2. [Soma]    Simplify soma-mcp to execution + experiment store tools only
3. [Nous]    Create project from Chatty foundation
4. [Nous]    Add knowledge base (moved from soma-memory)
5. [Nous]    Add graph versioning
6. [Nous]    Add evaluation lab (evolved from Chats)
7. [Nous]    Add Nous MCP server (knowledge + eval + versioning tools)
8. [Nous]    Connect to Soma MCP for execution
9. [Nous]    Connect to ChronosVector MCP for temporal storage
10. [Nous]   Reward modeling pipeline
```

## The Vision

A researcher opens Nous. They:

1. **Design** a reasoning graph in Agent Studio
2. **Configure** a Soma pipeline for data processing
3. **Run** experiments via Soma workers
4. **Evaluate** results in the Evaluation Lab
5. **Analyze** trajectories and detect breakthroughs
6. **Document** findings in the Knowledge Base
7. **Optimize** prompts and hyperparameters
8. **Version** every change linked to metrics
9. **Report** results with auto-generated documentation
10. **Iterate** guided by the agent's analysis

Three projects. One platform. Complete research automation.
