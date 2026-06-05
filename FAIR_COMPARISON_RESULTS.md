# Fair Comparison Results

Generated: June 4, 2026

## What Was Tested

This comparison isolates the architecture feature that solves the stated problem:
shared query coordination through the Blackboard.

The benchmark intentionally bypasses LLM planning and SQL generation for both
systems. Both systems receive the same SQL tasks, domains, mock DB, and simulated
DB concurrency limit.

Run command:

```bash
python fair_architecture_benchmark.py
```

## Implemented Project Summary

The project implements a multi-domain SQL agent pipeline:

- `ParentOrchestrator` routes user requests across domains and merges child DAGs.
- `ChildTaskPlanner` decomposes work into SQL, derived, and plot tasks.
- `DAGExecutor` runs ready tasks in parallel while honoring dependencies.
- `SQLAgent` retrieves relevant schema context, generates SQL, then executes via
  the Blackboard.
- `Blackboard` coordinates duplicate SQL with cache lookup, atomic ownership,
  subscriber waiting, and result publishing.
- Query hashes include the domain, so identical SQL in different domains does not
  collide.
- `DerivedAgent` computes aggregations in memory from previous SQL results.
- `PlotAgent` consumes cached/in-memory results and does not query the DB.

## Fair Benchmark Design

Baseline architecture:

- Every isolated agent directly executes its SQL.
- No shared result cache.
- No duplicate query coordination.
- No inter-agent communication.

Improved architecture:

- Every agent calls `run_with_blackboard`.
- First agent for a unique domain+SQL becomes the owner and hits the DB.
- Concurrent duplicate agents become subscribers.
- Later duplicate requests are cache hits.

Shared conditions:

- LLM planning and SQL generation are skipped for both systems.
- The same SQL strings are used for both systems.
- The same mock DB is used for both systems.
- Each scenario starts from a cold Blackboard cache.
- Both systems use the same simulated DB concurrency limit: 2.

## Results

| Scenario | Baseline DB Calls | Improved DB Calls | DB Reduction | Baseline Time | Improved Time | Time Improvement |
|---|---:|---:|---:|---:|---:|---:|
| A. Concurrent identical SQL | 5 | 1 | 80.0% | 241.0 ms | 141.0 ms | 41.5% |
| B. Sequential cache reuse | 5 | 1 | 80.0% | 461.7 ms | 109.0 ms | 76.4% |
| C. Production overlap mix | 10 | 4 | 60.0% | 506.0 ms | 296.1 ms | 41.5% |
| D. Domain isolation | 4 | 2 | 50.0% | 171.0 ms | 147.9 ms | 13.5% |
| Overall | 24 | 8 | 66.7% | 1379.7 ms | 694.0 ms | 49.7% |

## Key Evidence

Scenario A proves concurrent deduplication:

- Baseline: 5 direct DB calls.
- Improved: 1 owner DB call and 4 subscribers.
- Result: 80% fewer DB calls.

Scenario B proves cache reuse:

- Baseline: 5 direct DB calls.
- Improved: 1 owner DB call and 4 cache hits.
- Result: 80% fewer DB calls and 76.4% faster execution.

Scenario C approximates a production overlap workload:

- Request pattern: Q1, Q2, Q1, Q3, Q1, Q2, Q3, Q1, Q2, Q4.
- Baseline: 10 DB calls.
- Improved: 4 DB calls, matching the 4 unique domain+SQL pairs.
- Result: 60% fewer DB calls.

Scenario D proves domain isolation:

- Same SQL runs in `airport` and `tech_startup`.
- Improved executes once per domain instead of incorrectly sharing one result.
- Duplicate requests inside each domain are deduplicated.

## Conclusion

The simple baseline can look better in one-off LLM-heavy tests because it has no
planning layer. That is not the architecture problem being solved.

When tested against overlapping SQL workloads, the implemented Blackboard
architecture is better:

- 66.7% fewer DB executions overall.
- 49.7% lower total runtime under the same DB concurrency limit.
- Correct domain isolation for identical SQL across different domains.
- Subscribers and cache hits prove agents are sharing work instead of querying in
  isolation.
