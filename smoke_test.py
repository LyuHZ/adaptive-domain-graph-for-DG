"""Run core tests without OfficeHome data or a CLIP checkpoint."""

from __future__ import annotations

import inspect

import tests.test_core as core_tests


def main() -> None:
    tests = sorted(
        (name, function)
        for name, function in inspect.getmembers(core_tests, inspect.isfunction)
        if name.startswith("test_")
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"SMOKE_OK: {len(tests)} core tests passed")


if __name__ == "__main__":
    main()
