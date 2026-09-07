"""A local dashboard for the pricer: what is wired, and whether the feed answers.

    python dashboard/server.py           # then open http://127.0.0.1:8765

Lives outside ``torch_pricer`` on purpose. Nothing in the package imports it, it
adds no dependency to the library, and deleting this folder costs it nothing.

Runs entirely on this machine, and that is the point rather than a limitation.
The one question worth answering -- does the market-data connection work -- can
only be answered by a process holding the API key and able to reach the vendor,
so a hosted page could show the project's shape but never test it.

**Why FastAPI rather than the standard library.** The slow panels are the
interesting ones: a live chain fetch, and a test suite that takes ninety seconds.
Polling shows those as "running..." and then, eventually, an answer -- which is
the opposite of watching something work. Server-sent events stream each stage as
it completes, and ``EventSourceResponse`` handles the protocol, the keep-alives
and the client hanging up. The blocking work runs in a thread and pushes onto a
queue, so a long pytest run never blocks the event loop and the pulse keeps
ticking underneath it.

Bound to 127.0.0.1. The API key is read from the environment, never echoed into a
response, and never written to disk.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "index.html"
KEY_VARS = ("POLYGON_API_KEY", "MASSIVE_API_KEY")
sys.path.insert(0, str(ROOT))
# -- panels ---------------------------------------------------------------


def environment() -> dict:
    """Interpreter, torch, device, and whether a key is reachable."""
    out = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": str(ROOT),
        "key_set": [v for v in KEY_VARS if os.environ.get(v)],
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        out["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_mb": torch.cuda.get_device_properties(i).total_memory // 2**20,
                "free_mb": torch.cuda.mem_get_info(i)[0] // 2**20,
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # pragma: no cover - environment probe
        out["torch_error"] = str(exc)

    try:
        head = subprocess.run(
            ["git", "log", "-1", "--format=%h %s"], cwd=ROOT,
            capture_output=True, text=True, timeout=5,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5,
        )
        out["commit"] = head.stdout.strip()
        out["branch"] = branch.stdout.strip()
    except Exception:
        pass
    return out


def module_map() -> dict:
    """Every module, grouped by the method axis, with a line count and import check."""
    import importlib

    groups: dict[str, list] = {}
    for path in sorted((ROOT / "torch_pricer").rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        rel = path.relative_to(ROOT)
        parts = rel.parts[1:-1]
        group = "/".join(parts) if parts else "(root)"
        name = ".".join(rel.with_suffix("").parts)
        entry = {"module": name, "lines": len(path.read_text().splitlines())}
        try:
            importlib.import_module(name)
            entry["ok"] = True
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        groups.setdefault(group, []).append(entry)
    return {"groups": groups}


def connection(live: bool) -> dict:
    """Ask Massive for a chain, or explain precisely why we cannot."""
    from torch_pricer.data.massive import MassiveSource
    from torch_pricer.errors import PricerError

    key = next((v for v in KEY_VARS if os.environ.get(v)), None)
    if not live:
        return {"mode": "skipped", "key_var": key}
    if key is None:
        return {
            "mode": "no-key",
            "detail": (
                "Neither POLYGON_API_KEY nor MASSIVE_API_KEY is set in this "
                "process. Note that ~/.zshrc is not read by non-interactive "
                "shells -- put the export in ~/.zshenv, or launch this "
                "dashboard from an interactive shell."
            ),
        }

    started = time.time()
    try:
        chain = MassiveSource().fetch("I:SPX")
    except PricerError as exc:
        return {
            "mode": "failed", "key_var": key,
            "elapsed": round(time.time() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:
        return {
            "mode": "failed", "key_var": key,
            "elapsed": round(time.time() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-1500:],
        }

    return {
        "mode": "ok", "key_var": key,
        "elapsed": round(time.time() - started, 2),
        "ticker": chain.spot.ticker, "spot": chain.spot.value,
        "as_of": chain.as_of.isoformat(),
        "quotes": len(chain.options), "expiries": len(chain.expiries()),
        "first_expiries": [d.isoformat() for d in chain.expiries()[:6]],
    }


def pipeline_events(live: bool) -> Iterator[dict]:
    """Run the market-data pipeline, yielding each hop the moment it completes.

    A generator rather than a function returning a list: the live fetch and the
    SVI fit are the slow steps, and the point of streaming is to see the cheap
    stages land while those are still running.
    """
    from torch_pricer.calibration.inputs import CalibrationInputs
    from torch_pricer.calibration.svi_fit import fit_surface
    from torch_pricer.data import synthetic
    from torch_pricer.data.clean import clean, static_arbitrage
    from torch_pricer.data.implied import attach_implied_vols, otm_only
    from torch_pricer.data.massive import MassiveSource
    from torch_pricer.market.forward import curves_from_forwards, implied_forwards
    from torch_pricer.market.snapshot import MarketSnapshot
    from torch_pricer.market.svi import SVISlice, SVISurface

    emitted = []

    def stage(name, detail, ok=True, note=""):
        event = {"name": name, "detail": detail, "ok": ok, "note": note}
        emitted.append(event)
        return {"type": "stage", **event}

    try:
        if live:
            chain = MassiveSource().fetch("I:SPX")
            source = "Massive I:SPX (live)"
        else:
            as_of = dt.date.today()
            chain = synthetic.chain(
                as_of, [as_of + dt.timedelta(days=d) for d in (105, 196, 287)]
            )
            source = "synthetic (manufactured, r=4.2% q=1.3%)"
        yield stage("fetch", f"{len(chain.options)} quotes, {len(chain.expiries())} "
                       f"expiries, spot {chain.spot.value:,.2f}", note=source)

        cleaned, report = clean(chain)
        yield stage("clean", f"kept {report.kept}/{report.total}",
              ok=report.kept > 0,
              note=", ".join(f"{k}: {v}" for k, v in report.rejected.most_common()) or "nothing dropped")

        arb = static_arbitrage(cleaned)
        yield stage("static arbitrage",
              "clean" if arb.clean else
              f"{len(arb.monotonicity)} monotonicity, {len(arb.convexity)} convexity, "
              f"{len(arb.calendar)} calendar",
              ok=arb.clean,
              note="model-free: quotes checked against each other")

        yf = lambda e: (e - chain.as_of).days / 365.0  # noqa: E731
        fits = implied_forwards(cleaned, yf)
        yield stage("parity forwards", f"{len(fits)} of {len(chain.expiries())} expiries fitted",
              note="no rate curve, no dividend estimate -- option quotes only")

        discount, dividend = curves_from_forwards(fits, chain.spot.value)
        base = MarketSnapshot.flat(chain.as_of, spot=chain.spot.value,
                                   ticker=chain.spot.ticker)
        snap = type(base)(ticker=chain.spot.ticker, as_of=chain.as_of, spot=base.spot,
                          discount=discount, dividend=dividend,
                          vol_surface=base.vol_surface)
        import torch

        worst = max(
            abs(float(snap.forward(torch.tensor(f.t, dtype=torch.float64)).detach()) / f.forward - 1)
            for f in fits
        )
        yield stage("curves", f"reproduce every parity forward to {worst:.2e} relative",
              ok=worst < 1e-8,
              note="so log-moneyness is the market's own")

        with_vols, inv = attach_implied_vols(cleaned, fits)
        yield stage("implied vols", f"inverted {inv.inverted}/{inv.total}",
              ok=inv.inverted > 0,
              note=", ".join(f"{k}: {v}" for k, v in inv.failed.most_common()) or "all inverted")

        quotes = otm_only(with_vols, fits)
        yield stage("OTM filter", f"{len(with_vols.options)} -> {len(quotes.options)} quotes",
              note="the in-the-money leg is mostly intrinsic; its vega is small")

        surface = SVISurface(
            slices=[SVISlice(0.02, 0.08, -0.4, 0.0, 0.2, f.t) for f in fits],
            forward=snap.forward, as_of=chain.as_of,
        )
        result = fit_surface(surface, CalibrationInputs(market=snap, quotes=quotes))
        residual = float(result.residuals.abs().max())
        yield stage("SVI fit", f"{result.residuals.numel()} quotes, max {residual:.4%} vol",
              ok=residual < 0.05,
              note="residuals are in vol, the unit the market quotes")

        table = [
            {
                "expiry": f.expiry.isoformat(), "t": round(f.t, 4),
                "forward": round(f.forward, 2), "discount": round(f.discount, 6),
                "zero": round(f.zero_rate, 6), "pairs": f.n_pairs,
                "residual": f.max_residual,
                "atm_vol": float(surface.vol(
                    torch.tensor(f.forward, dtype=torch.float64), f.t).detach()),
            }
            for f in fits
        ]
        yield {"type": "done", "ok": True, "source": source, "forwards": table}
    except Exception as exc:
        yield stage("failed", f"{type(exc).__name__}: {exc}", ok=False)
        yield {"type": "done", "ok": False,
               "trace": traceback.format_exc()[-2000:]}


def engines() -> dict:
    """Price one contract every way the package can, and show the spread.

    An American put has no closed form, so agreement between a lattice, a PDE
    solver and a simulation that share no arithmetic is the strongest statement
    available about whether any of them is right.
    """
    import math

    from torch_pricer.instruments.spec import Right, Style, VanillaOption
    from torch_pricer.market.snapshot import MarketSnapshot
    from torch_pricer.models.black import BlackScholesModel
    from torch_pricer.pricer.analytic.black import black_price
    from torch_pricer.pricer.monte_carlo.engine import MCConfig, price
    from torch_pricer.pricer.pde.engine import price_pde
    from torch_pricer.pricer.tree.engine import price_lattice

    S, K, T, V, R, Q = 100.0, 100.0, 1.0, 0.20, 0.05, 0.0
    as_of, expiry = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
    market = MarketSnapshot.flat(as_of, spot=S, flat_rate=R, flat_dividend=Q, flat_vol=V)
    rows = []

    for style, right in ((Style.EUROPEAN, Right.PUT), (Style.AMERICAN, Right.PUT)):
        spec = VanillaOption(strike=K, maturity=expiry, right=right, style=style)
        entry = {"contract": f"{style.value} {right.value}"}

        if style is Style.EUROPEAN:
            fwd, disc = S * math.exp((R - Q) * T), math.exp(-R * T)
            entry["analytic"] = float(black_price(fwd, K, T, V, disc, -1.0))
        entry["tree"] = price_lattice(S, K, T, V, R, Q, right.value, style.value,
                                      n_steps=4096).price
        entry["pde"] = price_pde(S, K, T, V, R, Q, right.value, style.value,
                                 n_space=2048, n_time=1024).price
        mc = price(spec, market, BlackScholesModel(V),
                   MCConfig(n_paths=60_000, n_steps=100, seed=3, device="cpu"))
        entry["monte_carlo"] = mc.price
        entry["mc_stderr"] = mc.stderr

        values = [v for k, v in entry.items()
                  if k in ("analytic", "tree", "pde", "monte_carlo")]
        entry["spread"] = max(values) - min(values)
        rows.append(entry)
    return {"rows": rows}


def test_events() -> Iterator[dict]:
    """Run the suite on CPU, forwarding pytest's output as it is produced.

    ``-v`` rather than ``-q`` is what makes this stream at all. Quiet mode writes
    its progress as dots on a single line with no newline until the section ends,
    so iterating by line blocks for the whole run -- exactly the behaviour
    streaming exists to remove. Verbose mode emits one line per test, which
    arrives immediately and says which test is running. ``-u`` and
    ``PYTHONUNBUFFERED`` stop Python adding its own buffering on top.
    """
    started = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT),
               PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest", "tests/", "-v", "--tb=line",
         "-p", "no:cacheprovider", "--color=no"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        yield {"type": "line", "text": line,
               "elapsed": round(time.time() - started, 1)}
    proc.wait()
    # The summary pytest prints last is the line worth surfacing; in verbose mode
    # it is the "=== N passed in Ns ===" banner rather than the final test.
    banner = [ln for ln in lines if " passed" in ln or " failed" in ln or " error" in ln]
    yield {
        "type": "done",
        "elapsed": round(time.time() - started, 1),
        "returncode": proc.returncode,
        "summary": (banner[-1] if banner else "(no output)").strip("= "),
        "failures": [ln for ln in lines if ln.startswith("FAILED")],
    }




# -- transport ------------------------------------------------------------


def pulse() -> dict:
    """The cheap heartbeat: clock, device memory, key presence.

    Deliberately does no work beyond reading counters, so it can tick every two
    seconds without perturbing what it is reporting on.
    """
    out = {
        "time": dt.datetime.now().strftime("%H:%M:%S"),
        "key_set": [v for v in KEY_VARS if os.environ.get(v)],
    }
    try:
        import torch

        if torch.cuda.is_available():
            out["devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "free_mb": torch.cuda.mem_get_info(i)[0] // 2**20,
                    "total_mb": torch.cuda.get_device_properties(i).total_memory // 2**20,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return out


async def _stream(make: Callable[[], Iterator[dict]]):
    """Bridge a blocking generator to SSE without stalling the event loop.

    The work runs in a worker thread and pushes onto a queue; this coroutine
    drains it. Doing it the obvious way -- iterating the generator inline --
    would block the loop for the ninety seconds pytest takes, freezing the pulse
    and every other panel along with it.
    """
    events: queue.Queue = queue.Queue()
    sentinel = object()

    def run():
        try:
            for event in make():
                events.put(event)
        except Exception as exc:  # surface it on the page, not just in the log
            events.put({"type": "done", "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc()[-2000:]})
        finally:
            events.put(sentinel)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        event = await loop.run_in_executor(None, events.get)
        if event is sentinel:
            return
        yield {"data": json.dumps(event)}


app = FastAPI(
    title="torch_pricer dashboard",
    description="Local diagnostics: what is wired, and whether the feed answers.",
    version="1.0",
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PAGE)


@app.get("/api/pulse", summary="Clock, device memory, key presence")
def api_pulse():
    return pulse()


@app.get("/api/env", summary="Interpreter, torch, CUDA, git head")
def api_env():
    return environment()


@app.get("/api/map", summary="Every module, with an import check")
def api_map():
    return module_map()


@app.get("/api/connection", summary="Ask Massive for a chain")
def api_connection(live: bool = False):
    return connection(live)


@app.get("/api/engines", summary="Price one contract by every method")
def api_engines():
    return engines()


@app.get("/api/stream/pulse", summary="Heartbeat, server-pushed")
async def api_stream_pulse():
    async def ticker():
        while True:
            yield {"data": json.dumps(pulse())}
            await asyncio.sleep(2)

    return EventSourceResponse(ticker())


@app.get("/api/stream/pipeline", summary="Pipeline, streamed stage by stage")
async def api_stream_pipeline(live: bool = False):
    return EventSourceResponse(_stream(lambda: pipeline_events(live)))


@app.get("/api/stream/tests", summary="pytest, streamed line by line")
async def api_stream_tests():
    return EventSourceResponse(_stream(test_events))


def main() -> int:
    import uvicorn

    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    key = next((v for v in KEY_VARS if os.environ.get(v)), None)
    print(f"pricer dashboard -> http://127.0.0.1:{port}     (API docs at /docs)")
    print(f"API key: {key or 'not set in this process'}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
