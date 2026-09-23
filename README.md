# Delta

A minimal coding agent harness in Python, inspired by [pi](https://github.com/earendil-works/pi) (TypeScript). Delta runs a language model in a
loop with file and shell tools, saves every session so you can resume or branch it,
and ships both a terminal UI and a scriptable one-shot mode.

Requires Python 3.12 or newer.

## Install

The package name on PyPI is `deltaa`; once installed it gives you a `delta` command.

Fastest path — one script, works on a machine with nothing set up yet:

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/kartikg-gk/delta/main/scripts/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/kartikg-gk/delta/main/scripts/install.ps1 | iex
```

No admin/root privileges needed. If `uv` isn't found on your machine, the script
grabs it first, then puts Delta in its own isolated environment and runs
`delta --version` at the end so you know it worked. Read either script under
[`scripts/`](scripts/) before running it if you'd rather not pipe-to-shell blind.

If you already manage Python packages some other way, pick one:

```bash
uv tool install deltaa      # isolated, recommended if you have uv
pipx install deltaa         # isolated, no uv needed
python -m pip install deltaa
```

Confirm it landed:

```bash
delta --version
```

When a new version ships:

```bash
uv tool upgrade deltaa
```

## Quickstart

Delta needs a model provider. Set one up once:

```bash
delta config
```

Now `cd` into whatever project you want it working on and launch it:

```bash
cd my-project
delta
```

With no arguments, `delta` drops you into the terminal UI — start typing and hit
Enter, same as any chat interface:

```
what does the auth module in this repo do?
```

Need it non-interactively instead — for a script, a Makefile target, CI, whatever —
pass `-p` and it prints the answer and exits:

```bash
delta -p "list every route this API exposes"
delta --model gpt-5 -p "where is the retry logic for failed requests?"
```

If you'd rather stay in a plain scrolling terminal than the full-screen UI:

```bash
delta --repl
```

## Providers and configuration

Built-in providers: **OpenAI**, **Anthropic**, **OpenRouter**, and **Ollama** for local
models.

`delta config` walks you through picking a provider, a model, and an API key. It can
also run without prompts:

```bash
delta config --provider openai --model gpt-5
delta config show        # saved settings, keys masked
delta config path        # where the file lives
```

Settings are stored in `~/.delta/config.toml`, outside your project, with owner-only
permissions. Environment variables take priority over the file:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` | Provider credentials |
| `DELTA_API_KEY` | API key for `delta config` without an interactive prompt |
| `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` | Point a provider at another compatible endpoint |
| `DELTA_PROVIDER` / `DELTA_MODEL` | Force the active provider or model |
| `DELTA_HOME` | Move `~/.delta` (sessions, logs, skills, prompts, extensions) |
| `DELTA_CONFIG_DIR` | Move the config file |

Inside a session, `/model` and `/provider` switch on the fly.

## Commands

From your shell:

```bash
delta [prompt]                           # terminal UI, or a one-shot prompt
delta -p "..."                           # print mode: stream the answer, then exit
delta -r <session-id>                    # resume a saved session
delta config [show|path]                 # provider setup
delta provider list|setup|select <name>  # inspect or switch providers
delta session list                       # saved sessions
delta session export <id> [-f text|json|jsonl]
delta session stats [id]                 # token and cost totals
```

Useful flags: `--model/-m`, `--provider`, `--system-prompt`, `--max-turns`,
`--session-dir`, `--no-session`, `--repl`, `--verbose/-v`.

Inside a session, slash commands give live control:

| Area | Commands |
|---|---|
| Session | `/name` `/continue` `/export` `/stats` |
| History | `/compact` `/branch` `/branches` `/rewind` |
| Model | `/model` `/provider` `/think` `/plan` |
| Workspace | `/shell` `/skills` `/create-skill` `/reload` |
| Info | `/help` `/version` `/diag` `/quit` |

The terminal UI also has `/new` and `/session` for starting and switching sessions.

## Skills, prompts, and extensions

**Project instructions.** Delta reads `AGENTS.md` from, in increasing priority:
`~/.delta/`, `~/.agents/`, `<project>/.delta/`, `<project>/.agents/`, and the project
root. All files found are combined.

**Skills** are markdown instructions the agent can pull in when relevant. Put one at
`<project>/.delta/skills/<name>/SKILL.md` (or under `.agents/skills/`, or the same
paths in your home directory), list them with `/skills`, and invoke one directly with
`/skill:<name> [args]`. `/create-skill <name> <text>` writes a new one for you.

**Prompt templates** live in `.delta/prompts/` (project or home). Run one by typing
`/<template-name>`.

**Extensions** add capabilities without a restart: drop a `.py` file that defines a
`register` function into `~/.delta/extensions/` or `<project>/.delta/extensions/`,
then run `/reload`.

## Project structure

Three packages, layered so dependencies only point inward: `delta_harness` imports
neither of the others, `delta_model` imports the harness, and `delta_app` imports both.

```
src/
  delta_harness/          # core: the agent loop and its vocabulary
    contracts/            #   transcript entries, tool specs, event types, JSON values
    provider/             #   the model-provider interface and raw stream events
    session/              #   durable JSONL records, storage, replay, index, summaries
    engine.py             #   the agent loop
    driver.py             #   transcript-owning harness around the loop

  delta_model/            # providers: everything that talks to a model API
    claude.py  _claude/   #   Anthropic Messages API
    oai_compatible.py     #   OpenAI-compatible endpoints (OpenAI, OpenRouter, Ollama, ...)
    _oai/                 #   request building, response parsing, normalization, transport
    transport/            #   HTTP client, retry with backoff, error classification
    scripted.py           #   a deterministic fake provider for tests
    settings.py           #   environment-driven provider settings
    limits.py             #   context and output limits

  delta_app/              # application: the parts you actually use
    cli/                  #   entry point, argument parsing, `delta config`
    tui/                  #   terminal UI: transcript, composer, completions, pickers
    ui/render.py          #   headless output (plain, JSON, transcript)
    config/               #   config schema, TOML store, setup wizard, hidden key input
    tools/                #   file tools, shell tool, shared helpers
    runtime.py            #   session construction shared by every frontend
    conversation.py       #   the session object the CLI and UI both drive
    directives.py         #   slash-command registry and dispatch
    instructions.py       #   system-prompt assembly
    skillset.py           #   markdown skills
    prompts.py            #   prompt templates
    discovery.py          #   where skills, prompts, and sessions are found
    plugins.py            #   file-based extensions with hot reload
    refresh.py            #   `/reload` wiring
    context/budget.py     #   token accounting and compaction thresholds
    safety.py             #   tool approval and output scrubbing
    planning.py           #   plan mode
    reasoning.py          #   thinking modes
    sessions.py           #   create, resume, list, export
    logs.py               #   per-session run logs

tests/                    # one test module per concern
scripts/                  # install scripts
```

## Development

```bash
git clone https://github.com/kartikg-gk/delta.git
cd delta
uv sync
uv run delta --version
uv run pytest
uv run ruff check .
```

To point the global `delta` command at your checkout instead of a PyPI release:

```bash
uv tool install --editable --force .
```

Code changes in the checkout take effect right away with this setup. What doesn't
update on its own is the tool's dependency list and version metadata — those are
snapshotted at install time, so after pulling changes that touch `pyproject.toml`
(a new dependency, a version bump), rerun the command above to pick them up.

## License

MIT
