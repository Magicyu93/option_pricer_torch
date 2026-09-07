# Dashboard

    python dashboard/server.py          # http://127.0.0.1:8765   (docs at /docs)

Local diagnostics for the pricer: what is wired, and whether the market-data
feed answers. Outside `torch_pricer` on purpose — nothing in the package imports
it, it adds no dependency to the library, and deleting this folder costs it
nothing.

**Launch it from an interactive shell.** The connection panel needs
`POLYGON_API_KEY`, and `~/.zshrc` is not read by non-interactive shells — put the
export in `~/.zshenv` if you want it visible everywhere.

Six panels: environment, market-data connection, the full pipeline stage by
stage, cross-method price agreement, the test suite, and a per-module import
check.

Built on FastAPI, because the interesting panels are the slow ones. A live chain
fetch and a ninety-second test run are worth *watching*; polling would show
"running…" and then an answer. Server-sent events push each stage as it lands,
and the blocking work runs in a worker thread feeding a queue, so pytest never
stalls the event loop and the heartbeat keeps ticking underneath it.
