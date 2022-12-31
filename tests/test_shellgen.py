import io
import re

import pytest

from pycheribenchplot.core.shellgen import ScriptBuilder
from pycheribenchplot.core.task import DataFileTarget


def test_shellgen_base(fake_simple_benchmark):
    """
    Check that the script builder generates the correct sequence of commands
    in each section.
    """
    script = ScriptBuilder(fake_simple_benchmark)
    target = DataFileTarget.from_benchmark(fake_simple_benchmark, "test-file")

    # Check that we generated the correct number of sections
    assert len(script.sections["benchmark"]) == 2  # number of iterations in the fixture config

    script.sections["pre-benchmark"].add_cmd("test-pre", ["foo", "bar"])
    script.sections["post-benchmark"].add_cmd("test-post", ["100"], env={"KEY": "VALUE"})
    script.sections["last"].add_cmd("test-last")
    script.benchmark_sections[0]["pre-benchmark"].add_cmd("test-pre-i0", cpu=2)
    script.benchmark_sections[1]["benchmark"].add_cmd("test-i1", output=target)
    # Verify output
    result = io.StringIO()
    script.generate(result)

    result.seek(0)
    expect = [
        r"#!/bin/sh", r"test-pre foo bar", r"cpuset -c -l 2 test-pre-i0",
        r"test-i1 >> /root/benchmark-output/test-file", r"KEY=VALUE test-post 100", r"test-last"
    ]
    expect_next = 0
    for line in result:
        if re.match(expect[expect_next], line):
            expect_next += 1
    assert expect_next == len(expect), f"Last matching position {expect[expect_next]}"
