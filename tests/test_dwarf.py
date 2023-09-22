import math
from pathlib import Path

import numpy as np
import pytest

from pycheribenchplot.core.dwarf import (DWARFManager, DWARFStructLayoutModel, GraphConversionVisitor)
from pycheribenchplot.ext import pydwarf


@pytest.fixture
def dwarf_manager(fake_simple_benchmark):
    return DWARFManager(fake_simple_benchmark)


def check_member_tflags(df_row, *args):
    """
    Helper to check structure member type flags
    """
    flags = df_row["member_flags"]
    if not args:
        assert flags == pydwarf.TypeInfoFlags.kNone, f"Invalid flags kNone != {flags}"
    check = [getattr(pydwarf.TypeInfoFlags, k).value for k in args]

    all_flags = 0
    for name, check_flag in zip(args, check):
        assert (flags & check_flag) != 0, f"Flag {name} is not set"
        all_flags |= check_flag
    complement = flags & ~all_flags
    assert complement == 0, f"Extra flags set {complement}"


def check_member(df, name):
    """
    Grab the row corresponding to a member with the given name.
    The row must be unique and it is returned as a series.
    """
    df = df.reset_index()
    result = df[df["member_name"] == name]
    assert len(result) > 0, f"No member matching member_name == {name}"
    assert len(result) == 1, f"Too many rows matching member_name == {name}"
    return result.iloc[0]


def test_register_obj(dwarf_manager):
    dw = dwarf_manager.register_object("k", "tests/assets/riscv_purecap_test_dwarf_simple")
    assert dw is not None
    assert dw.path == "tests/assets/riscv_purecap_test_dwarf_simple"


def test_extract_struct_layout(dwarf_manager, fake_session):
    """
    Check that we detect simple structure layouts correctly.
    """
    dw = dwarf_manager.register_object("k", "tests/assets/riscv_purecap_test_dwarf_simple")

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)

    foo = df.xs("foo", level="base_name").reset_index()
    assert (foo["size"] == 48).all()
    assert (foo["file"].map(lambda p: Path(p).name) == "test_dwarf_simple.c").all()
    assert (foo["line"] == 9).all()

    bar = df.xs("bar", level="base_name").reset_index()
    assert (bar["size"] == 8).all()
    assert (foo["file"].map(lambda p: Path(p).name) == "test_dwarf_simple.c").all()
    assert (bar["line"] == 4).all()


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members", 8)])
def test_extract_layout_members(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Test the type layout flattening interface.
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 15
    a = check_member(foo, "a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    b = check_member(foo, "b")
    assert b["member_type_name"] == "char *"
    assert b["member_size"] == ptr_size
    c = check_member(foo, "c")
    assert c["member_type_name"] == "struct bar"
    assert c["member_size"] == 8
    cx = check_member(foo, "c.x")
    assert cx["member_type_name"] == "int"
    assert cx["member_size"] == 4
    cy = check_member(foo, "c.y")
    assert cy["member_type_name"] == "int"
    assert cy["member_size"] == 4
    d = check_member(foo, "d")
    assert d["member_type_name"] == "char const *"
    assert d["member_size"] == ptr_size
    e = check_member(foo, "e")
    assert e["member_type_name"] == "char * const"
    assert e["member_size"] == ptr_size
    f = check_member(foo, "f")
    assert f["member_type_name"] == "void volatile const *"
    assert f["member_size"] == ptr_size
    g = check_member(foo, "g")
    assert g["member_type_name"] == "int * *"
    assert g["member_size"] == ptr_size
    h = check_member(foo, "h")
    assert h["member_type_name"] == "int * [10]"
    assert h["member_size"] == 10 * ptr_size
    i = check_member(foo, "i")
    assert i["member_type_name"] == "void(int, int) *"
    assert i["member_size"] == ptr_size
    j = check_member(foo, "j")
    assert j["member_type_name"] == "int [10] *"
    assert j["member_size"] == ptr_size
    k = check_member(foo, "k")
    assert k["member_type_name"] == "baz_t"
    assert k["member_size"] == 8
    kz = check_member(foo, "k.z")
    assert kz["member_type_name"] == "long"
    assert kz["member_size"] == 8
    l = check_member(foo, "l")
    assert l["member_type_name"] == "struct forward *"
    assert l["member_size"] == ptr_size

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 2
    x = check_member(bar, "x")
    y = check_member(bar, "y")
    assert x["member_type_name"] == "int"
    assert y["member_type_name"] == "int"

    baz = df.xs("baz", level="base_name")
    assert len(baz) == 1
    z = check_member(baz, "z")
    assert z["member_type_name"] == "long"


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members", 8)])
def test_extract_layout_members_graph(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Test the conversion of the typeinfo container to a graph
    """
    fake_session.user_config.src_path = Path.cwd()
    expect_file_path = "tests/assets/test_dwarf_struct_members.c"
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)

    # Count only nodes we have control over
    nodes = [n for n in g.nodes if n.file.find("test_dwarf_struct_members") != -1]
    assert len(nodes) == 21

    def check_node(nid):
        if nid not in nodes:
            for n in nodes:
                print("check n=", n, "MATCH:", nid == n)
        return nid in nodes

    def check_size(nid, size_purecap, size_hybrid):
        if ptr_size == 16:
            assert g.nodes[nid]["size"] == size_purecap
        else:
            assert g.nodes[nid]["size"] == size_hybrid

    NodeID = GraphConversionVisitor.NodeID

    # Verify node information
    foo = NodeID(file=expect_file_path, line=15, base_name="foo", member_name=None, member_offset=0)
    assert check_node(foo)
    assert g.nodes[foo]["type_name"] == "struct foo"
    check_size(foo, 0x150, 0xa8)
    assert g.in_degree(foo) == 0
    assert g.out_degree(foo) == 12

    foo_a = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="a", member_offset=0)
    assert check_node(foo_a)
    assert g.nodes[foo_a]["type_name"] == "int"
    check_size(foo_a, 4, 4)
    assert g.in_degree(foo_a) == 1
    assert g.out_degree(foo_a) == 0

    foo_b = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="b", member_offset=ptr_size)
    assert check_node(foo_b)
    assert g.nodes[foo_b]["type_name"] == "char *"
    check_size(foo_b, ptr_size, ptr_size)
    assert g.in_degree(foo_b) == 1
    assert g.out_degree(foo_b) == 0

    foo_c = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="c", member_offset=2 * ptr_size)
    assert check_node(foo_c)
    assert g.nodes[foo_c]["type_name"] == "struct bar"
    check_size(foo_c, 8, 8)
    assert g.in_degree(foo_c) == 1
    assert g.out_degree(foo_c) == 2

    foo_cx = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="x", member_offset=2 * ptr_size)
    assert check_node(foo_cx)
    assert g.nodes[foo_cx]["type_name"] == "int"
    check_size(foo_cx, 4, 4)
    assert g.in_degree(foo_cx) == 1
    assert g.out_degree(foo_cx) == 0

    foo_cy = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="y", member_offset=2 * ptr_size + 4)
    assert check_node(foo_cy)
    assert g.nodes[foo_cy]["type_name"] == "int"
    check_size(foo_cy, 4, 4)
    assert g.in_degree(foo_cy) == 1
    assert g.out_degree(foo_cy) == 0

    foo_d = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="d", member_offset=3 * ptr_size)
    assert check_node(foo_d)
    assert g.nodes[foo_d]["type_name"] == "char const *"
    check_size(foo_d, ptr_size, ptr_size)
    assert g.in_degree(foo_d) == 1
    assert g.out_degree(foo_d) == 0

    foo_e = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="e", member_offset=4 * ptr_size)
    assert check_node(foo_e)
    assert g.nodes[foo_e]["type_name"] == "char * const"
    check_size(foo_e, ptr_size, ptr_size)
    assert g.in_degree(foo_e) == 1
    assert g.out_degree(foo_e) == 0

    foo_f = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="f", member_offset=5 * ptr_size)
    assert check_node(foo_f)
    assert g.nodes[foo_f]["type_name"] == "void volatile const *"
    check_size(foo_f, ptr_size, ptr_size)
    assert g.in_degree(foo_f) == 1
    assert g.out_degree(foo_f) == 0

    foo_g = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="g", member_offset=6 * ptr_size)
    assert check_node(foo_g)
    assert g.nodes[foo_g]["type_name"] == "int * *"
    check_size(foo_g, ptr_size, ptr_size)
    assert g.in_degree(foo_g) == 1
    assert g.out_degree(foo_g) == 0

    foo_h = NodeID(file=expect_file_path, line=15, base_name="foo", member_name="h", member_offset=7 * ptr_size)
    assert check_node(foo_h)
    assert g.nodes[foo_h]["type_name"] == "int * [10]"
    check_size(foo_h, 10 * ptr_size, 10 * ptr_size)
    assert g.in_degree(foo_h) == 1
    assert g.out_degree(foo_h) == 0

    foo_i = NodeID(file=expect_file_path,
                   line=15,
                   base_name="foo",
                   member_name="i",
                   member_offset=7 * ptr_size + 10 * ptr_size)
    assert check_node(foo_i)
    assert g.nodes[foo_i]["type_name"] == "void(int, int) *"
    check_size(foo_i, ptr_size, ptr_size)
    assert g.in_degree(foo_i) == 1
    assert g.out_degree(foo_i) == 0

    foo_j = NodeID(file=expect_file_path,
                   line=15,
                   base_name="foo",
                   member_name="j",
                   member_offset=8 * ptr_size + 10 * ptr_size)
    assert check_node(foo_j)
    assert g.nodes[foo_j]["type_name"] == "int [10] *"
    check_size(foo_j, ptr_size, ptr_size)
    assert g.in_degree(foo_j) == 1
    assert g.out_degree(foo_j) == 0

    foo_k = NodeID(file=expect_file_path,
                   line=15,
                   base_name="foo",
                   member_name="k",
                   member_offset=9 * ptr_size + 10 * ptr_size)
    assert check_node(foo_k)
    assert g.nodes[foo_k]["type_name"] == "baz_t"
    check_size(foo_k, 8, 8)
    assert g.in_degree(foo_k) == 1
    assert g.out_degree(foo_k) == 1

    foo_kz = NodeID(file=expect_file_path,
                    line=15,
                    base_name="foo",
                    member_name="z",
                    member_offset=9 * ptr_size + 10 * ptr_size)
    assert check_node(foo_kz)
    assert g.nodes[foo_kz]["type_name"] == "long"
    check_size(foo_kz, 8, 8)
    assert g.in_degree(foo_kz) == 1
    assert g.out_degree(foo_kz) == 0

    foo_l = NodeID(file=expect_file_path,
                   line=15,
                   base_name="foo",
                   member_name="l",
                   member_offset=10 * ptr_size + 10 * ptr_size)
    assert check_node(foo_l)
    assert g.nodes[foo_l]["type_name"] == "struct forward *"
    check_size(foo_l, ptr_size, ptr_size)
    assert g.in_degree(foo_l) == 1
    assert g.out_degree(foo_l) == 0


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members_anon", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members_anon", 8)])
def test_extract_layout_members_anon(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check the extraction of layouts with anonymous structures and unions.
    """
    fake_session.user_config.src_path = Path.cwd()
    expect_file_path = "tests/assets/test_dwarf_struct_members_anon.c"
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 8
    a = check_member(foo, "a")
    assert a["member_type_name"] == "baz_t"
    assert a["member_size"] == 8
    az = check_member(foo, "a.z")
    assert az["member_type_name"] == "long"
    assert az["member_size"] == 8

    a = check_member(foo, f"<anon>.{ptr_size}")
    assert a["member_type_name"] == f"struct <anon>.{expect_file_path}.10"
    assert a["member_size"] == 2 * ptr_size
    assert a["member_offset"] == ptr_size
    s_a = check_member(foo, f"<anon>.{ptr_size}.s_a")
    assert s_a["member_type_name"] == "int"
    assert s_a["member_size"] == 4
    assert s_a["member_offset"] == ptr_size
    s_b = check_member(foo, f"<anon>.{ptr_size}.s_b")
    assert s_b["member_type_name"] == "char *"
    assert s_b["member_size"] == ptr_size
    assert s_b["member_offset"] == 2 * ptr_size

    un = check_member(foo, f"<anon>.{3 * ptr_size}")
    assert un["member_type_name"] == f"union <anon>.{expect_file_path}.14"
    assert un["member_size"] == ptr_size
    assert un["member_offset"] == 3 * ptr_size
    un_a = check_member(foo, f"<anon>.{3 * ptr_size}.un_a")
    assert un_a["member_type_name"] == "int"
    assert un_a["member_size"] == 4
    assert un_a["member_offset"] == 3 * ptr_size
    un_b = check_member(foo, f"<anon>.{3 * ptr_size}.un_b")
    assert un_b["member_type_name"] == "char *"
    assert un_b["member_size"] == ptr_size
    assert un_b["member_offset"] == 3 * ptr_size

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 3
    nested = check_member(bar, f"nested")
    assert nested["member_type_name"] == f"struct <anon>.{expect_file_path}.21"
    assert nested["member_size"] == 2 * ptr_size
    assert nested["member_offset"] == 0
    a = check_member(bar, "nested.a")
    assert a["member_type_name"] == "long"
    assert a["member_size"] == 8
    assert a["member_offset"] == 0
    b = check_member(bar, "nested.b")
    assert b["member_type_name"] == "char *"
    assert b["member_size"] == ptr_size
    assert b["member_offset"] == ptr_size


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_nested", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_nested", 8)])
def test_extract_layout_members_nesting(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check the extraction of layouts with deeply nested structures and union combinations.
    """
    fake_session.user_config.src_path = Path.cwd()
    expect_file_path = "tests/assets/test_dwarf_nested.c"
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 12
    a = check_member(foo, "a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    assert a["member_offset"] == 0

    b = check_member(foo, "b")
    assert b["member_type_name"] == f"union <anon>.{expect_file_path}.19"
    assert b["member_size"] == 16
    assert b["member_offset"] == 8

    bar = check_member(foo, "b.b_bar")
    assert bar["member_type_name"] == f"struct bar"
    assert bar["member_size"] == 16
    assert bar["member_offset"] == 8

    bar_x = check_member(foo, "b.b_bar.x")
    assert bar_x["member_type_name"] == "int"
    assert bar_x["member_size"] == 4
    assert bar_x["member_offset"] == 8

    bar_u = check_member(foo, "b.b_bar.u")
    assert bar_u["member_type_name"] == "union nested_union"
    assert bar_u["member_size"] == 8
    assert bar_u["member_offset"] == 16

    bar_ux = check_member(foo, "b.b_bar.u.x")
    assert bar_ux["member_type_name"] == "int"
    assert bar_ux["member_size"] == 4
    assert bar_ux["member_offset"] == 16

    bar_uy = check_member(foo, "b.b_bar.u.y")
    assert bar_uy["member_type_name"] == "long"
    assert bar_uy["member_size"] == 8
    assert bar_uy["member_offset"] == 16

    baz = check_member(foo, "b.b_baz")
    assert baz["member_type_name"] == f"struct baz"
    assert baz["member_size"] == 16
    assert baz["member_offset"] == 8

    baz_v = check_member(foo, "b.b_baz.v")
    assert baz_v["member_type_name"] == "int"
    assert baz_v["member_size"] == 4
    assert baz_v["member_offset"] == 8

    baz_u = check_member(foo, "b.b_baz.u")
    assert baz_u["member_type_name"] == "union nested_union"
    assert baz_u["member_size"] == 8
    assert baz_u["member_offset"] == 16

    baz_ux = check_member(foo, "b.b_baz.u.x")
    assert baz_ux["member_type_name"] == "int"
    assert baz_ux["member_size"] == 4
    assert baz_ux["member_offset"] == 16

    baz_uy = check_member(foo, "b.b_baz.u.y")
    assert baz_uy["member_type_name"] == "long"
    assert baz_uy["member_size"] == 8
    assert baz_uy["member_offset"] == 16


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_empty_struct", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_empty_struct", 8)])
def test_extract_empty_struct(fake_session, dwarf_manager, asset_file, ptr_size):
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 1
    assert (bar.index.get_level_values("member_name") == "<empty>").all()
    assert bar["member_size"].isna().all()
    assert bar["member_type_name"].isna().all()

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 3
    a = check_member(foo, "a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    assert a["member_offset"] == 0
    b = check_member(foo, "b")
    assert b["member_type_name"] == "struct bar"
    assert b["member_size"] == 0
    assert b["member_offset"] == 4
    be = check_member(foo, "b.<empty>")
    assert be[["member_type_name"]].isna().all()
    assert be[["member_size"]].isna().all()
    assert be["member_offset"] == 4


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_zero_length_array", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_zero_length_array", 8)])
def test_zero_length_array(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check that zero-length array behave correctly
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    def align_up(v):
        return ((v + (ptr_size - 1)) & ~(ptr_size - 1))

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 9
    why_l = check_member(foo, "why_l_")
    assert why_l["member_type_name"] == "char [0]"
    assert why_l["member_size"] == 0
    assert why_l["member_offset"] == 0
    why = check_member(foo, "why")
    assert why["member_type_name"] == "char const *"
    assert why["member_size"] == ptr_size
    assert why["member_offset"] == 0
    why_r = check_member(foo, "why_r_")
    assert why_r["member_type_name"] == "char [0]"
    assert why_r["member_size"] == 0
    assert why_r["member_offset"] == ptr_size

    nargs_l = check_member(foo, "nargs_l_")
    assert nargs_l["member_type_name"] == "char [0]"
    assert nargs_l["member_size"] == 0
    assert nargs_l["member_offset"] == ptr_size
    nargs = check_member(foo, "nargs")
    assert nargs["member_type_name"] == "int"
    assert nargs["member_size"] == 4
    assert nargs["member_offset"] == ptr_size
    nargs_r = check_member(foo, "nargs_r_")
    assert nargs_r["member_type_name"] == "char [12]"
    assert nargs_r["member_size"] == 12
    assert nargs_r["member_offset"] == 4 + ptr_size

    args_l = check_member(foo, "args_l_")
    assert args_l["member_type_name"] == "char [0]"
    assert args_l["member_size"] == 0
    assert args_l["member_offset"] == align_up(16 + ptr_size)
    args = check_member(foo, "args")
    assert args["member_type_name"] == "void * *"
    assert args["member_size"] == ptr_size
    assert args["member_offset"] == align_up(16 + ptr_size)
    args_r = check_member(foo, "args_r_")
    assert args_r["member_type_name"] == "char [0]"
    assert args_r["member_size"] == 0
    assert args_r["member_offset"] == align_up(16 + 2 * ptr_size)


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_bitfield_struct", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_bitfield_struct", 8)])
def test_extract_bitfield_struct(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check that bitfields are counted correctly.

    Note that the parse_struct_layout() data model does not care about bit-level resolution.
    Instead, bitfileds that belong to the same word/memeber will show up as aliased members
    of an union.
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 5

    before = check_member(foo, "before")
    assert before["member_offset"] == 0
    assert before["member_size"] == 1
    bitfield_a = check_member(foo, "bitfield_a")
    assert bitfield_a["member_offset"] == 0
    assert bitfield_a["member_size"] == 4
    bitfield_b = check_member(foo, "bitfield_b")
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_size"] == 4
    after = check_member(foo, "after")
    assert after["member_offset"] == 7
    assert after["member_size"] == 1
    x = check_member(foo, "x")
    assert x["member_offset"] == 8
    assert x["member_size"] == 8

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 4
    before = check_member(bar, "before")
    assert before["member_offset"] == 0
    assert before["member_size"] == 4
    bitfield_a = check_member(bar, "bitfield_a")
    assert bitfield_a["member_offset"] == 4
    assert bitfield_a["member_size"] == 4
    bitfield_b = check_member(bar, "bitfield_b")
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_size"] == 4
    x = check_member(bar, "x")
    assert x["member_offset"] == 8
    assert x["member_size"] == 8


@pytest.mark.user_config
@pytest.mark.slow
def test_extract_kernel_struct(fake_session, dwarf_manager, benchplot_user_config):
    kernel_path = benchplot_user_config.sdk_path / "rootfs-riscv64-purecap/boot/kernel/kernel.full"
    dw = dwarf_manager.register_object("k", kernel_path)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique
