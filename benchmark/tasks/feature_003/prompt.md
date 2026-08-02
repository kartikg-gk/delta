`docs/design/result-cache.md` describes a caching layer we agreed on but never built. The module `cache.py` exists as an empty shell with the intended public functions stubbed out.

Implement it to match the design note.

Requirements:

- Follow the document — where it specifies behaviour, the behaviour is not negotiable.
- Where the document is genuinely silent, choose the simplest option and note the decision in your final summary.
- Do not change the public function signatures already stubbed in `cache.py`; other modules import them.
- Include tests for the eviction and invalidation behaviour.
