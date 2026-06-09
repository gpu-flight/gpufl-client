"""Targeted multi-pass profiling for long embedded jobs.

A multi-day training job can't be re-run end-to-end per engine, and CUPTI
runs only ONE profiling engine per process (re-init is unsupported). So a
multi-engine deep-dive of a hot scope is **N short runs — one engine each,
sharing an analysis_id** — orchestrated as separate PROCESSES, not several
engines in one live ``with`` block.

You name the engines in ONE place — the launcher — which spawns one process
per engine and auto-assigns the ``analysis_id`` + ``pass_index`` +
``pass_count`` + engine (via the ``GPUFL_*`` env vars the C++ core reads)::

    gpufl trace --passes Trace,RangeProfiler,PcSampling -- python train.py

Your script stays **engine-agnostic** — ``targeting()`` inherits its assigned
engine and analysis-group labels from the environment; you only describe the
WINDOW (which scope, how many iterations)::

    import gpufl

    with gpufl.targeting("attn_block", iters=20, warmup=5) as run:
        for batch in run.steps(loader):     # auto-stops after warmup+iters
            train_step(batch)

Each launcher pass re-runs the same script with a different engine, bounded
to the window, and the dashboard's Analysis Group view merges them for the
"attn_block" scope — no ``pass_index`` / ``pass_count`` bookkeeping in your
code. (Hot scope early in training → the bounded window keeps each pass
short; a scope deep in a long run wants a checkpoint loaded just before it.)

**Standalone (no launcher)** — pass ``engine=`` explicitly (otherwise the run
is Monitor-only, no kernel profiling); to self-group several runs without the
launcher, pass a shared ``analysis_id=`` and a distinct ``pass_index=`` per
run. Prefer the launcher, which does all of that for you.

When gpufl is disabled (``GPUFL_DISABLED=1``, or init failed / no GPU) the run
degrades to a pass-through: every item is yielded, no scope is opened, and the
window NEVER truncates the loop — toggling gpufl off leaves training unchanged.
"""

from contextlib import contextmanager


class TargetingRun:
    """Handle yielded by :func:`gpufl.targeting`. Brackets each step's hot
    region with the target scope and closes the profiling window after
    ``warmup + iters`` measured steps.

    Use :meth:`steps` to wrap an iterable (auto-stops the loop), or
    :meth:`step` per iteration for hand-written loops (check :attr:`done`).
    """

    def __init__(self, scope, tag, iters, warmup, scope_factory, enabled=True):
        self._scope = scope
        self._tag = tag
        # iters=None → no auto-stop: scope every step until the loop ends.
        self._iters = iters
        self._warmup = warmup or 0
        self._scope_factory = scope_factory  # injected gpufl.Scope (no import cycle)
        self._enabled = enabled
        self._measured = 0
        self._warmups_done = 0
        self.done = False

    @contextmanager
    def step(self):
        """Bracket one step's hot region with the target scope.

        Warmup steps open a ``<scope>_warmup`` sub-scope (excluded from the
        measured window); measured steps open ``<scope>``. After the window
        closes (or when disabled), this is a zero-overhead pass-through that
        opens no scope and never counts — so trailing iterations are free
        and the loop is never truncated when gpufl is off.
        """
        if not self._enabled or self.done:
            yield self
            return
        in_warmup = self._warmups_done < self._warmup
        name = f"{self._scope}_warmup" if in_warmup else self._scope
        with self._scope_factory(name, self._tag):
            yield self
        if in_warmup:
            self._warmups_done += 1
        else:
            self._measured += 1
            if self._iters is not None and self._measured >= self._iters:
                self.done = True

    def steps(self, iterable):
        """Yield items from ``iterable``, bracketing each with :meth:`step`,
        and STOP once the window closes (``warmup + iters`` steps). Use it as
        the loop iterable so the run self-bounds::

            for batch in run.steps(loader):
                train_step(batch)
        """
        for item in iterable:
            with self.step():
                yield item
            if self.done:
                return

    @property
    def measured(self):
        """Number of measured (non-warmup) steps profiled so far."""
        return self._measured


@contextmanager
def targeting(scope, *, iters=20, warmup=3, tag="",
              engine=None, analysis_id=None, pass_index=None, pass_count=None,
              **kwargs):
    """Profile a targeted scope as ONE pass of a multi-pass analysis.

    Context manager. Composes :func:`gpufl.session` (init + shutdown + optional
    upload) with a bounded per-step scope window. **Normally run under the
    launcher** — ``gpufl trace --passes Trace,RangeProfiler,PcSampling --
    python train.py`` — which assigns the engine and the analysis-group labels
    per pass; then this call only describes the WINDOW. See the module
    docstring for the full recipe.

    Args:
        scope: Name of the hot region (the merge key — use the SAME name for
            every pass of one analysis group).
        iters: Measured steps to profile before auto-stopping the window.
            None → no auto-stop (scope every step until the loop ends). Keep it
            IDENTICAL across the passes of one analysis so the per-scope
            Execution Signature matches and the backend merges them.
        warmup: Cold-start steps run before the measured window (bracketed in a
            ``<scope>_warmup`` sub-scope, excluded from the merge).
        tag: Optional scope category tag.

        engine / analysis_id / pass_index / pass_count: **Launcher-set; omit
            them under ``gpufl trace --passes``.** Provide them only for a
            STANDALONE run (no launcher): ``engine=`` picks this run's engine
            (else Monitor-only), and a shared ``analysis_id=`` + distinct
            ``pass_index=`` per run self-group several runs without the
            launcher.
        **kwargs: Forwarded to :func:`gpufl.init` (``app_name``, ``log_path``,
            ``backend_url``, ``api_key``, ``enable_stack_trace``, …).

    Yields:
        A :class:`TargetingRun` — call ``run.steps(iterable)`` (auto-stops) or
        ``run.step()`` per iteration.
    """
    # Lazy import to avoid an import cycle (gpufl.__init__ imports this module
    # at load time; Scope/session exist by the time this is CALLED).
    from . import Scope, session

    # engine=None → leave it to the launcher's GPUFL_PROFILING_ENGINE env
    # override (applied inside the C++ init), so the script is engine-agnostic.
    if engine is not None:
        kwargs['profiling_engine'] = engine

    with session(analysis_id=analysis_id, pass_index=pass_index,
                 pass_count=pass_count, **kwargs) as ok:
        yield TargetingRun(scope, tag, iters, warmup, Scope, enabled=bool(ok))
