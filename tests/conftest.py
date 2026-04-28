import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runperf",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.performance (timing / throughput).",
    )


def pytest_collection_modifyitems(config, items):
    env_on = os.environ.get("RUN_PERF", "").strip().lower() in ("1", "true", "yes")
    if config.getoption("--runperf") or env_on:
        return
    skip_perf = pytest.mark.skip(
        reason="pass --runperf or set RUN_PERF=1 to run performance tests"
    )
    for item in items:
        if item.get_closest_marker("performance"):
            item.add_marker(skip_perf)
