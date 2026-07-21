"""Solar-UniBench experiments package.

Each experiment ships as ``e{n}_<topic>.py`` exposing
``run(loader, out_dir) -> dict`` (figures + numeric metrics). The orchestrator
(:mod:`scripts.run_experiments`) discovers them lazily so the harness runs
incrementally as experiments land.
"""
