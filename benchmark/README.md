# Delta benchmark suite

29 task-level benchmarks for the Delta coding agent.

Each task directory holds `prompt.md` (the verbatim instruction sent to Delta), `repo/` (the fixture checkout Delta works in), `verify.py` (the automated grader), and `metadata.json`.

## Running

```bash
python benchmark/run.py --task bug_fix_001 --model <model-id>
python benchmark/run.py --all --model <model-id> --out results.jsonl
```

The runner copies `repo/` to a scratch directory, starts Delta there with `prompt.md` as the opening message, waits for the run to end, then executes `verify.py` against the scratch copy. Recorded per task: pass/fail, turns, tool calls, tokens in/out, wall-clock seconds, cost.

## Tasks (7 Easy / 13 Medium / 9 Hard)

| Task | Difficulty | Tags | Evaluates |
|---|---|---|---|
| `bug_fix_001` | Easy | bug-fix, debugging, navigation | Baseline loop competence: read a traceback, follow it to the real file, make a minimal edit. |
| `bug_fix_002` | Easy | bug-fix, debugging | Off-by-one reasoning from a behavioural report rather than a stack trace. |
| `bug_fix_003` | Medium | bug-fix, debugging | Tracing a runtime TypeError back through two call layers to a data boundary (parsing) rather than patching at the crash site. |
| `bug_fix_004` | Medium | bug-fix, debugging | Diagnosing a shared-mutable-state bug that only manifests on the second call. |
| `bug_fix_005` | Hard | bug-fix, performance, debugging, terminal | Performance debugging with no exception to follow. |
| `feature_001` | Easy | feature, cli | Smallest end-to-end feature: locate the arg parser, add a flag, branch the output path, keep the default intact. |
| `feature_002` | Medium | feature, api, testing | Implementing a resilience policy correctly: classifying retriable vs terminal errors, honouring Retry-After, and — critically — writing tests that don't actually sleep. |
| `feature_003` | Medium | feature, docs-driven | Documentation-driven implementation: the spec exists, the code doesn't. |
| `feature_004` | Hard | feature, multi-file, api | Full vertical slice across model, storage, serialisation, routing, and tests. |
| `feature_005` | Medium | feature, multi-file, observability | Threading a cross-cutting concern through an existing codebase without touching every call site. |
| `refactor_001` | Easy | refactor, multi-file | Recognising near-identical-but-not-identical code. |
| `refactor_002` | Hard | refactor, multi-file, navigation | Large-scale mechanical change with a hard compatibility constraint. |
| `refactor_003` | Hard | refactor, multi-file, performance | Async conversion — the classic place agents go wrong by leaving a blocking call inside a coroutine or by firing requests sequentially with await in a loop. |
| `multifile_001` | Medium | multi-file, refactor, navigation | Exhaustive search-and-replace where naive text replacement fails: the old name appears as a substring of unrelated identifiers, in strings, and in serialised fixtures that must NOT change. |
| `multifile_002` | Medium | multi-file, feature, api | Vertical change through model, migration, serialiser, API, and client. |
| `terminal_001` | Medium | terminal, config, debugging | Pure terminal-loop work: run a failing command, read the error, adjust config, run again. |
| `terminal_002` | Hard | terminal, debugging, git | Git archaeology under a bounded budget. |
| `terminal_003` | Easy | terminal, config | Mechanical but tool-dependent: the agent must actually run the linters to discover the violations rather than eyeballing files. |
| `debug_001` | Hard | debugging, testing | Non-deterministic failure. |
| `debug_002` | Hard | debugging, performance | Resource-leak diagnosis. |
| `debug_003` | Hard | debugging, bug-fix | No crash, no failing test — only a description of wrong output. |
| `api_001` | Medium | api, feature, docs-driven | Reading an external contract (OpenAPI document) and writing a typed client against it, including auth and error mapping, with no live service to poke at. |
| `api_002` | Hard | api, refactor, multi-file, docs-driven | Cross-cutting dependency migration where the new API is shaped differently, not just renamed. |
| `config_001` | Easy | config, terminal | Minimal terminal-and-config task. |
| `config_002` | Medium | config, feature | Implementing layered configuration precedence correctly, including the boundary cases (empty string vs unset, type coercion) that naive implementations get wrong. |
| `docs_001` | Medium | docs-driven, feature | Spec adherence with zero implementation hints. |
| `docs_002` | Medium | docs-driven, navigation, terminal | Reverse direction: verify docs against code. |
| `nav_001` | Easy | navigation, read-only | Pure comprehension with no edit. |
| `nav_002` | Medium | navigation, multi-file, refactor | Exhaustive discovery under adversarial conditions: the helper is also reached via an alias and a dynamic lookup, so grep alone under-reports. |
