# Multi-Domain Orchestrator Redesign

This plan details the implementation of a parent-child orchestrator architecture to support multiple databases with overlapping schemas and data.

## Goal Description
The objective is to handle complex analytical queries that span across multiple distinct databases/domains (e.g., `airport`, `tech_startup`, `restaurant`). These domains will share some overlapping schemas (like `employees`) but contain different domain-specific mock data. A Parent Orchestrator will route tasks to the appropriate Child Orchestrators, which will then generate domain-specific execution DAGs.

## User Review Required
> [!IMPORTANT]
> **Domain-specific FAISS Indexes**: I plan to create separate FAISS indexes for each domain (e.g., `schema_index/airport/index.faiss`) rather than one massive index with domain metadata tags. This ensures strong isolation and avoids LLM confusion.
> 
> **DAG Merging**: The Parent Orchestrator will invoke the Child Orchestrators in parallel and simply merge their returned task DAGs (prefixing task IDs to avoid collisions, e.g., `t1` -> `airport_t1`). The DAG Executor will run the combined graph. Plot Agents at the end will be able to synthesize data across domains because all results share the blackboard. Is this approach acceptable?

## Open Questions
> [!WARNING]
> Do you want the Parent Orchestrator to generate *cross-domain* derived/plot tasks? Right now, if a user asks to "compare tech startup salaries to airport salaries", Child 1 gets the tech salaries, Child 2 gets the airport salaries. I plan to let the DAG executor's PlotAgent or a Parent Plot Task combine the results from the Blackboard.

## Proposed Changes

---

### Database & Schema Mocking
#### [MODIFY] [schema/indexer.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/schema/indexer.py)
* Replace `MOCK_SCHEMA` with a dictionary of domains: `{"airport": [...], "tech_startup": [...], "restaurant": [...]}`.
* Include an `employees` table in all domains with domain-specific fields (e.g., `clearance_level` for airport, `programming_language` for tech).
* Update `build_index` to loop through the domains and create separate FAISS indexes in `schema_index/<domain>/index.faiss`.

#### [MODIFY] [schema/retriever.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/schema/retriever.py)
* Update `SchemaRetriever` to accept a `domain` parameter.
* Implement a `get_retriever(domain: str)` factory function to load the specific FAISS index for the requested domain.

#### [MODIFY] [db/pool.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/db/pool.py)
* Update `execute_query(sql, domain)` to accept the domain.
* Update `_mock_execute` to return different datasets depending on the domain (especially for the overlapping `employees` table).

---

### Planning Architecture
#### [MODIFY] [planner/task_planner.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/planner/task_planner.py)
* Add `domain: str = "default"` to the `TaskNode` dataclass.
* Rename `GlobalTaskPlanner` to `ChildTaskPlanner`. Update it to accept a `domain` string during initialization and assign that domain to all `TaskNode`s it creates.

#### [NEW] [planner/parent_planner.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/planner/parent_planner.py)
* Create `ParentOrchestrator`.
* Add an LLM call that takes the user's request and returns a list of relevant domains (from `["airport", "tech_startup", "restaurant"]`) and an optional global plot/comparison task.
* Instantiate `ChildTaskPlanner` for each identified domain, gather their DAGs, and prefix task IDs (e.g., `airport_t1`) to merge them into a single global execution DAG.

---

### Execution Layer
#### [MODIFY] [agents/sql_agent.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/agents/sql_agent.py)
* Update initialization to accept `domain`.
* Fetch the domain-specific schema retriever.
* Update `query_hash` generation to include the `domain` (e.g., `hash(domain + normalized_sql)`). This ensures a query for `SELECT * FROM employees` in the airport domain does not cache-hit the same query in the tech domain.

#### [MODIFY] [dag/executor.py](file:///c:/Users/Arshman%20Khawar/Documents/SQL%20Agents%20Task/dag/executor.py)
* Update `_run_sql_task` to pass the `task.domain` to the `SQLAgent`.

## Verification Plan

### Automated Tests
* Create `test_multi_domain.py` to execute a cross-domain query: *"Fetch employee details for the tech startup and the airport, and give me a comparison plot."*
* Assert that:
  1. The `ParentOrchestrator` correctly identifies `["tech_startup", "airport"]`.
  2. The `DAGExecutor` runs `SQLAgent` instances for both domains.
  3. The `query_hash` isolation works correctly (no cache collision between the domains).
  4. The Plot Agent successfully aggregates the results from the Blackboard.
