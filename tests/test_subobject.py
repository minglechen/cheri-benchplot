from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from test_dwarf import check_member

from pycheribenchplot.core.dwarf import DWARFManager, GraphConversionVisitor
from pycheribenchplot.subobject.imprecise_subobject import (AllImpreciseMembersPlot, ExtractImpreciseSubobject,
                                                            ExtractImpreciseSubobjectConfig, RequiredSubobjectPrecision)

NodeID = GraphConversionVisitor.NodeID


@pytest.fixture
def extract_task(fake_simple_benchmark):
    return ExtractImpreciseSubobject(fake_simple_benchmark, None,
                                     ExtractImpreciseSubobjectConfig(dwarf_data_sources=[]))


@pytest.fixture
def plot_imprecise_task(fake_session):
    return AllImpreciseMembersPlot(fake_session, None)


@pytest.fixture
def plot_precision_task(fake_session):
    return RequiredSubobjectPrecision(fake_session, None)


@pytest.fixture
def dwarf_manager(fake_simple_benchmark):
    return DWARFManager(fake_simple_benchmark)


@pytest.fixture
def expect_file_path():
    repo = Path.cwd().name
    expect_file_path = Path(repo) / "tests" / "assets" / "test_unrepresentable_subobject.c"
    return str(expect_file_path)


def test_find_imprecise(dwarf_manager, extract_task, expect_file_path):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)
    extract_task._find_imprecise(g)

    # Filter nodes to only the ones relevent to us
    nodes = [n for n in g.nodes if n.file.find("test_unrepresentable_subobject") != -1]
    assert len(nodes) == 21

    def check_node(nid):
        if nid not in nodes:
            for n in nodes:
                print("check n=", n, "MATCH:", nid == n)
        return nid in nodes

    def check_noalias(nid):
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_groups"]

    def check_aliasing(nid, groups: list[int] | int):
        if type(groups) == int:
            groups = [groups]
        assert set(g.nodes[nid]["alias_groups"]) == set(groups)

    def check_imprecise(nid, group: int, base: int, top: int):
        assert g.nodes[nid]["alias_group_id"] == group
        assert g.nodes[nid]["alias_aligned_base"] == base
        assert g.nodes[nid]["alias_aligned_top"] == top

    def check_not_imprecise(nid):
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_group_id"]
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_aligned_top"]
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_aligned_base"]

    # Verify node information for the test_large_subobject struct
    large = NodeID(file=expect_file_path, line=2, base_name="test_large_subobject", member_name=None, member_offset=0)
    assert check_node(large)
    assert g.nodes[large]["type_name"] == "struct test_large_subobject"
    assert g.nodes[large]["has_imprecise"]
    check_not_imprecise(large)
    check_noalias(large)

    skew_offset = NodeID(file=expect_file_path,
                         line=2,
                         base_name="test_large_subobject",
                         member_name="skew_offset",
                         member_offset=0)
    assert check_node(skew_offset)
    assert g.nodes[skew_offset]["type_name"] == "int"
    check_not_imprecise(skew_offset)
    check_aliasing(skew_offset, 0)  # aliasing large_buffer
    large_buffer = NodeID(file=expect_file_path,
                          line=2,
                          base_name="test_large_subobject",
                          member_name="large_buffer",
                          member_offset=4)
    check_imprecise(large_buffer, 0, 0, 8192)
    check_noalias(large_buffer)

    # Verify node information for the test_complex struct
    nested = NodeID(file=expect_file_path, line=20, base_name="test_complex", member_name=None, member_offset=0)
    assert check_node(nested)
    assert g.nodes[large]["has_imprecise"]
    assert g.nodes[nested]["type_name"] == "struct test_complex"
    check_not_imprecise(nested)
    check_noalias(nested)

    before = NodeID(file=expect_file_path, line=20, base_name="test_complex", member_name="before", member_offset=0)
    assert check_node(before)
    assert g.nodes[before]["type_name"] == "int"
    check_not_imprecise(before)
    check_aliasing(before, [0, 1])  # aliasing inner and inner.buf_before
    inner = NodeID(file=expect_file_path, line=20, base_name="test_complex", member_name="inner", member_offset=4)
    assert check_node(inner)
    assert g.nodes[inner]["type_name"] == f"struct <anon>.{expect_file_path}.22"
    check_imprecise(inner, 0, 0, 16384)
    check_noalias(inner)
    inner_buf_before = NodeID(file=expect_file_path,
                              line=20,
                              base_name="test_complex",
                              member_name="buf_before",
                              member_offset=4)
    assert check_node(inner_buf_before)
    assert g.nodes[inner_buf_before]["type_name"] == f"char [8188]"
    check_imprecise(inner_buf_before, 1, 0, 8192)
    check_noalias(inner_buf_before)
    inner_buf_after = NodeID(file=expect_file_path,
                             line=20,
                             base_name="test_complex",
                             member_name="buf_after",
                             member_offset=8192)
    assert check_node(inner_buf_after)
    assert g.nodes[inner_buf_after]["type_name"] == f"char [8187]"
    check_imprecise(inner_buf_after, 2, 8192, 16384)
    check_noalias(inner_buf_after)
    after = NodeID(file=expect_file_path, line=20, base_name="test_complex", member_name="after", member_offset=16379)
    assert check_node(after)
    assert g.nodes[after]["type_name"] == f"char [10]"
    check_not_imprecise(after)
    check_aliasing(after, [0, 2])  # aliasing inner and inner.buf_after

    ## Check the age_softc_layout regression
    # This uses a similar layout as the age_softc in the if_age.ko kernel module
    age_softc_layout = NodeID(file=expect_file_path,
                              line=14,
                              base_name="test_age_softc_layout",
                              member_name=None,
                              member_offset=0)
    assert check_node(age_softc_layout)
    assert g.nodes[age_softc_layout]["has_imprecise"]
    age_cdata = NodeID(file=expect_file_path,
                       line=14,
                       base_name="test_age_softc_layout",
                       member_name="cdata",
                       member_offset=0x250)
    assert check_node(age_cdata)
    check_imprecise(age_cdata, 0, 0x240, 0x240 + 0x6140 + 0x20)
    check_noalias(age_cdata)


def test_graph_io(dwarf_manager, extract_task, tmp_path):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)
    extract_task._find_imprecise(g)

    GraphConversionVisitor.dump(g, tmp_path / "test_dump.json")
    g2 = GraphConversionVisitor.load(tmp_path / "test_dump.json")
    for n in g2.nodes:
        assert type(n) == NodeID
    assert g.graph["roots"] == g2.graph["roots"]
    assert nx.utils.nodes_equal(g, g2)
    assert nx.utils.edges_equal(g, g2)

    GraphConversionVisitor.dump(g, tmp_path / "test_dump.json.gz")
    g2 = GraphConversionVisitor.load(tmp_path / "test_dump.json.gz")
    for n in g2.nodes:
        assert type(n) == NodeID
    assert g.graph["roots"] == g2.graph["roots"]
    assert nx.utils.nodes_equal(g, g2)
    assert nx.utils.edges_equal(g, g2)


def test_graph_extract_imprecise_members(dwarf_manager, extract_task, plot_imprecise_task):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)
    extract_task._find_imprecise(g)
    g.graph["dataset_gid"] = str(extract_task.benchmark.g_uuid)

    df = plot_imprecise_task._collect_imprecise_members([g])

    assert len(df) == 5

    large = df[df.index.get_level_values("base_name") == "test_large_subobject"]
    assert len(large) == 1
    assert (large.index.get_level_values("member_name") == "large_buffer").all()

    age = df[df.index.get_level_values("base_name") == "test_age_softc_layout"]
    assert len(age) == 1
    assert (age.index.get_level_values("member_name") == "cdata").all()

    cplx = df[df.index.get_level_values("base_name") == "test_complex"]
    assert len(cplx) == 3
    assert set(cplx.index.get_level_values("member_name")) == {"buf_before", "buf_after", "inner"}


def test_subobject_precision(fake_simple_benchmark, plot_precision_task):
    """
    Check that the precision calculation makes sense.
    Note that we are testing the RISC-V variant
    """
    riscv_id = fake_simple_benchmark.g_uuid
    assert plot_precision_task._compute_precision(0x00000000, 0x00100000) == 1
    assert plot_precision_task._compute_precision(0x0FFFFFFF, 0x10000000) == 1
    assert plot_precision_task._compute_precision(0x00000004, 0x00001004) == 11
    assert plot_precision_task._compute_precision(0x00000FFF, 0x00002001) == 13

    for top_shift in range(12):
        top = 1 << top_shift
        assert plot_precision_task._compute_platform_precision(
            riscv_id, 0x0000, top_shift) == 12, f"Platform precision for 1 << {top_shift} does not match"
    assert plot_precision_task._compute_platform_precision(riscv_id, 0x0000, 0x1000) == 10

    assert plot_precision_task._compute_platform_precision(riscv_id, 0x0FFF, 0x1000) == 12
    assert plot_precision_task._compute_platform_precision(riscv_id, 0x0FFF, 0x2001) == 10
