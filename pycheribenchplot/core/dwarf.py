import gzip
import json
import re
import subprocess
import typing
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field, fields, replace
from enum import Flag, IntFlag, auto
from functools import reduce
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandera as pa
from elftools.elf.elffile import ELFFile
from pandera.typing import DataFrame, Index, Series
from sortedcontainers import SortedDict

from ..ext import pydwarf
from .model import DataModel
from .util import gzopen, new_logger


def extract_at_str(die, at_name, default=None):
    """Helper to extract a string attribute from a DIE"""
    try:
        at = die.attributes[f"DW_AT_{at_name}"]
        return at.value.decode("ascii")
    except KeyError:
        return default


def extract_at_int(die, at_name, default=-1):
    """Helper to extract a string attribute from a DIE"""
    try:
        at = die.attributes[f"DW_AT_{at_name}"]
        return at.value
    except KeyError:
        return default


def extract_at_name(die, default=None):
    """Helper to extract a name from a DIE"""
    return extract_at_str(die, "name", default)


def extract_at_size(die, default=0):
    """Helper to extract the size attributes from a DIE"""
    return extract_at_int(die, "byte_size", default)


def extract_at_fileline(die, debug_line):
    src_file_off = extract_at_int(die, "decl_file", None)
    src_line = extract_at_int(die, "decl_line", None)
    if not src_file_off or not src_line:
        return (None, None)
    # decl_file is 1-based
    src_file_entry = debug_line["file_entry"][src_file_off - 1]
    src_file = src_file_entry["name"].decode("ascii")
    return (Path(src_file), src_line)


class DWARFDataRegistry:
    """
    Top level container for the data we mine from DWARF.
    This is passed around during the visit passes to collect
    and cross-reference the data.
    This is part of the main interface exposed to clients of this module.
    """
    def __init__(self, ef, dw, logger, arch_pointer_size):
        self.ef = ef
        self.dw = dw
        self.logger = logger
        self.arch_ptr_size = arch_pointer_size
        self.is_little_endian = self.ef.little_endian
        # map structure info by DIE offset
        self.struct_by_offset = {}
        # map structure info by unique key
        self.struct_by_key = {}
        # map typedef targets by DIE offset
        self.typedef_by_offset = {}
        # map struct offsets to callbacks for delayed fixup of fields that
        # would require recursively traversing structures.
        self.struct_ref_fixup = defaultdict(list)

    def get_struct_info(self, dedup=True) -> pd.DataFrame:
        """
        Export structure descriptors as pandas dataframe.
        If dedup is true, duplicate structs are removed.
        """
        valid_si = filter(lambda si: not si.is_anon, self.struct_by_key.values())
        si_frames = map(lambda si: si.to_frame(), valid_si)
        self.logger.debug("Build structure info dataframe")
        df = pd.concat(si_frames)
        df = df.set_index(["name", "src_file", "src_line", "member_name"])
        if dedup and not df.index.is_unique:
            dup = df.index.duplicated(keep="first")
            df = df[~dup]
        return df.sort_index()


@dataclass
class DWARFStructInfo:
    # Name of the structure or None if anonymous
    name: typing.Optional[str]
    # Structure size
    size: int
    # Compilation unit where we found the definition
    from_path: Path
    # Source file
    src_file: Path
    # Source line
    src_line: int
    # List of struct members descriptors
    members: typing.List["DWARFMemberInfo"] = field(default_factory=list)
    # Is the struct anonymous?
    is_anon: bool = False

    def to_frame(self):
        records = [m.to_frame_record() for m in self.members]
        if not records:
            # Add a single dummy row with no member name
            empty_record = {"name": np.nan, "offset": np.nan, "size": 0, "pad": 0}
            empty_record.update(DWARFTypeInfo().to_frame_record())
            records = [empty_record]
        df = pd.DataFrame.from_records(records)
        df = df.add_prefix("member_")
        df["name"] = self.name
        df["size"] = self.size
        df["from_path"] = self.from_path
        df["src_file"] = self.src_file
        df["src_line"] = self.src_line
        return df

    def apply_typedef(self, typedef: "DWARFTypeDef"):
        """Replace anonymous struct info with the typedef'ed info"""
        assert self.is_anon
        self.name = typedef.name
        self.src_file = typedef.src_file
        self.src_line = typedef.src_line
        self.is_anon = False

    @property
    def key(self):
        if self.is_anon:
            return ("anon", self.size, self.src_file, self.src_line)
        else:
            return (self.name, self.size, self.src_file, 0)

    def __str__(self):
        return f"{self.name} {self.src_file}:{self.src_line} @ {self.from_path}"


@dataclass
class DWARFMemberInfo:
    """
    Helper to wrap a struct member information
    """
    # member name, autogenerated <anon>.offset if anonymous
    name: str
    # type information
    type_info: "DWARFType"
    # byte offset in the structure
    offset: int
    # bit offset from byte offset for bitfields
    bit_offset: int = 0
    # byte size of the member
    size: int = None
    # bit size for bitfields
    bit_size: int = 0
    # byte padding to the next member
    pad: int = 0
    # bit padding to the next member for bitfields
    bit_pad: int = 0

    def __post_init__(self):
        assert self.type_info.size is not None
        if self.size is None:
            self.size = self.type_info.size

    def to_frame_record(self) -> dict:
        omit_fields = ["type_info"]
        record = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in omit_fields}
        type_record = {f"{k}": v for k, v in self.type_info.to_frame_record().items()}
        record.update(type_record)
        return record

    def __str__(self):
        value = f"{self.name} ({self.offset}:{self.bit_offset}) {self.type_info}"
        return value


@dataclass
class DWARFTypeDef:
    """Helper to hold information from a typedef"""
    name: str
    src_file: Path
    src_line: int

    def __str__(self):
        return f"typedef {self.name}"


@dataclass
class DWARFTypeInfo:
    """
    New class for holding type information
    """
    base_name: str = "<invalid>"
    name: str = ""
    size: int = 0
    # typedef name
    alias_name: str = None
    # Where it is defined, for typedefs and struct/union/enum
    src_file: Path = None
    src_line: int = None
    # Trailing internal padding, if applicable
    pad: int = 0
    # DIE offset for reference
    offset: int = None
    # Function parameters
    params: list = None
    # Array size if type is_array
    array_items: int = None
    # Flags
    is_typedef: bool = False
    is_ptr: bool = False
    is_struct: bool = False
    is_union: bool = False
    is_enum: bool = False
    is_const: bool = False
    is_volatile: bool = False
    is_array: bool = False

    def set_flag(self, name, value=True):
        if hasattr(self, name) and name.startswith("is_"):
            setattr(self, name, value)
        else:
            raise AttributeError(f"No DWARFTypeInfo flag {name}")

    def clear_flags(self):
        for f in fields(self):
            if f.name.startswith("is_"):
                setattr(self, f.name, False)

    def to_frame_record(self):
        record = {f"type_{k}": v for k, v in self.__dict__.items()}
        return record


class DWARFTypeResolver:
    """
    This is intended to be used internally to produce a DWARFTypeInfo describing
    a type.
    """
    def __init__(self, visitor, ctx):
        self.visitor = visitor
        self.ctx = ctx

    @property
    def debug_line(self):
        return self.visitor.current_debug_line

    def resolve_die(self, die) -> DWARFTypeInfo:
        chain = []
        self._follow_die_chain(die, chain)
        assert len(chain) > 0, "DIE resolved to nothing"
        # Now traverse the chain in reverse and fill the type info
        ti = DWARFTypeInfo()
        for die in reversed(chain):
            self._do_resolve_die(ti, die)
        if ti.is_struct:
            self._try_resolve_struct_pad(ti)
        return ti

    def _follow_die_chain(self, die, chain):
        try:
            t_die = die.get_DIE_from_attribute("DW_AT_type")
        except KeyError:
            # signal void
            chain.append(None)
            return

        chain.append(t_die)
        if (t_die.tag == "DW_TAG_base_type" or t_die.tag == "DW_TAG_subroutine_type"
                or t_die.tag == "DW_TAG_structure_type" or t_die.tag == "DW_TAG_union_type"
                or t_die.tag == "DW_TAG_enumeration_type"):
            return
        elif (t_die.tag == "DW_TAG_pointer_type" or t_die.tag == "DW_TAG_const_type"
              or t_die.tag == "DW_TAG_volatile_type" or t_die.tag == "DW_TAG_array_type"
              or t_die.tag == "DW_TAG_typedef"):
            self._follow_die_chain(t_die, chain)
        else:
            raise ValueError("Type resolver did not handle %s", t_die.tag)

    def _try_resolve_struct_pad(self, ti):
        """
        Attempt to resolve the trailing padding for a structure.
        If this is not possible, queue the TypeInfo for late resolution.
        """
        try:
            sinfo = self.ctx.struct_by_offset[ti.offset]
            if sinfo.members:
                ti.pad = sinfo.members[-1].pad
        except KeyError:
            # We can not recurse and visit again, so we keep a record of the
            # entries that need padding adjustment
            def callback(sinfo):
                if not sinfo.members:
                    return
                ti.pad = sinfo.members[-1].pad

            self.ctx.struct_ref_fixup[ti.offset].append(callback)

    def _do_resolve_die(self, ti, die):
        if die is None:
            ti.base_name = "void"
            ti.name = "void"
            return
        if die.tag == "DW_TAG_base_type":
            self._resolve_base_type(ti, die)
        elif die.tag == "DW_TAG_pointer_type":
            self._resolve_ptr_type(ti, die)
        elif die.tag == "DW_TAG_const_type":
            self._resolve_qualifier("const", ti, die)
        elif die.tag == "DW_TAG_volatile_type":
            self._resolve_qualifier("volatile", ti, die)
        elif die.tag == "DW_TAG_array_type":
            self._resolve_array_type(ti, die)
        elif die.tag == "DW_TAG_subroutine_type":
            self._resolve_fn_type(ti, die)
        elif die.tag == "DW_TAG_typedef":
            self._resolve_typedef(ti, die)
        elif (die.tag == "DW_TAG_structure_type" or die.tag == "DW_TAG_union_type"
              or die.tag == "DW_TAG_enumeration_type"):
            self._resolve_complex_type(ti, die)
        else:
            raise ValueError("Type resolver did not handle %s", die.tag)

    def _resolve_base_type(self, ti, die):
        name = extract_at_name(die)
        size = extract_at_size(die, None)
        assert size is not None, "No size for base type"
        ti.base_name = name
        ti.name = name
        ti.size = size

    def _resolve_ptr_type(self, ti, die):
        size = extract_at_size(die, self.ctx.arch_ptr_size)
        ti.name = ti.name + " *"
        ti.clear_flags()
        ti.set_flag("is_ptr")
        ti.size = size

    def _resolve_qualifier(self, qual, ti, die):
        ti.name = ti.name + f" {qual}"
        if qual == "volatile":
            ti.set_flag("is_volatile")
        elif qual == "const":
            ti.set_flag("is_const")

    def _resolve_array_type(self, ti, die):
        s_die = None
        for child in die.iter_children():
            if child.tag == "DW_TAG_subrange_type":
                s_die = child
                break
        assert s_die is not None
        lb = extract_at_int(s_die, "lower_bound", None)
        if lb is not None:
            raise ValueError("Unhandled DW_AT_lower_bound")

        count = extract_at_int(s_die, "count", None)
        ub = extract_at_int(s_die, "upper_bound", None)
        if count is not None:
            nitems = count
        elif ub is not None:
            nitems = ub + 1
        else:
            nitems = 0
        size_str = nitems or ""

        ti.name = ti.name + f" [{size_str}]"
        ti.set_flag("is_array")
        ti.array_items = nitems
        ti.size = ti.size * nitems

    def _resolve_fn_type(self, ti, die):
        ret_type = self.resolve_die(die)
        params = [ret_type]
        for child in die.iter_children():
            if child.tag == "DW_TAG_formal_parameter":
                param_type = self.resolve_die(child)
                params.append(param_type)
        param_str = ",".join(map(lambda p: p.name, params[1:]))

        ti.name = f"{ret_type.name}({param_str})" + ti.name
        ti.size = 0

    def _resolve_typedef(self, ti, die):
        name = extract_at_name(die)
        if not ti.is_typedef:
            # update alias name
            ti.alias_name = ti.name + (ti.alias_name or "")
        ti.set_flag("is_typedef")
        ti.name = name

    def _resolve_complex_type(self, ti, die):
        name = extract_at_name(die, "<anon>")
        size = extract_at_size(die)

        if die.tag == "DW_TAG_structure_type":
            ti.set_flag("is_struct")
            name_prefix = "struct"
        elif die.tag == "DW_TAG_union_type":
            ti.set_flag("is_union")
            name_prefix = "union"
        elif die.tag == "DW_TAG_enumeration_type":
            ti.set_flag("is_enum")
            name_prefix = "enum"
        else:
            assert False, "Not reached"
        ti.size = size
        ti.base_name = name
        ti.name = f"{name_prefix} {name}"
        ti.src_file, ti.src_line = extract_at_fileline(die, self.debug_line)
        ti.offset = die.offset


class DWARFVisitor:
    def __init__(self, ctx):
        self.ctx = ctx
        self.type_resolver = None
        self.current_unit = None
        self.current_unit_path = None
        self.current_debug_line = None

    def visit_unit(self, unit, die):
        self.current_unit = unit
        self.current_unit_path = Path(die.get_full_path())
        self.current_debug_line = self.ctx.dw.line_program_for_CU(unit)
        self.ctx.logger.debug("DWARF %s visit %s", self, self.current_unit_path)
        self.type_resolver = DWARFTypeResolver(self, self.ctx)

    def visit_structure_type(self, die):
        pass

    def visit_union_type(self, die):
        pass

    def visit_typedef(self, die):
        pass

    def __str__(self):
        return self.__class__.__name__


class StructInfoVisitor(DWARFVisitor):
    """
    Handle generation of the structure information from the DWARF tree.
    This will fill the `DataRegistry.struct_by_offset`, `DataRegistry.struct_by_key`
    and `DataRegistry.typedef_by_offset` fields.
    """
    def _resolve_type(self, member_name, die):
        try:
            return self.type_resolver.resolve_die(die)
        except Exception:
            self.ctx.logger.error("Error while handling member %s", member_name)
            raise

    def _resolve_nested_anon(self, prefix, offset, die):
        """Resolve members from a nested anonymous struct/union"""
        members = self._resolve_members(die)
        for m in members:
            m.offset += offset
            m.name = f"{prefix}.{m.name}"
        return members

    def _resolve_members(self, die):
        members = []
        for child in die.iter_children():
            if child.tag != "DW_TAG_member":
                continue
            offset = extract_at_int(child, "data_member_location", 0)
            name = extract_at_name(child, f"<anon>.{offset}")
            # if this is an anonymous struct/union, we traverse it and add its members
            # directly as members of the current struct/union
            member_die = child.get_DIE_from_attribute("DW_AT_type")
            member_type_name = extract_at_name(member_die)
            if member_die.tag == "DW_TAG_structure_type" and member_type_name is None:
                members.extend(self._resolve_nested_anon(name, offset, member_die))
            else:
                type_info = self._resolve_type(name, child)
                member = DWARFMemberInfo(name, type_info, offset)
                self._resolve_bitfield(child, member)
                members.append(member)
        self._resolve_members_padding(die, members)
        return members

    def _resolve_bitfield(self, die, member):
        """
        Check if a member die contains a bitfield and resolve the bitfield size/offset
        """
        bit_offset = extract_at_int(die, "bit_offset", 0)
        bit_size = extract_at_int(die, "bit_size", None)
        if bit_size is None:
            return
        if self.ctx.is_little_endian:
            bit_offset = member.type_info.size * 8 - (bit_offset + bit_size)
        assert bit_offset >= 0
        member.bit_offset = bit_offset
        member.size = int(np.floor(bit_size / 8))
        member.bit_size = bit_size % 8

    def _resolve_members_padding(self, die, members):
        # compute member paddings, assume there is never leading padding in a struct
        # sort to make sure ordering is reliable
        members = sorted(members, key=lambda m: m.offset * 8 + m.bit_offset)
        prev_off = extract_at_size(die) * 8
        for m in reversed(members):
            curr_off = m.offset * 8 + m.bit_offset
            if m.bit_size != 0:
                # XXX maybe test for None to distinguish non-bitfields for sure
                total_pad = prev_off - (curr_off + m.bit_size)
            else:
                total_pad = prev_off - (curr_off + m.size * 8)
            assert total_pad >= 0
            m.pad, m.bit_pad = int(total_pad / 8), total_pad % 8
            prev_off = curr_off
        return members

    def visit_structure_type(self, die):
        at_decl = die.attributes.get("DW_AT_declaration", None)
        if at_decl:
            # Skip non-defining entries
            return

        src_file, src_line = extract_at_fileline(die, self.current_debug_line)
        if not src_file or not src_line:
            raise ValueError("Missing file and line attributes")

        size = extract_at_size(die, None)
        if size is None:
            raise ValueError("Unexpected DWARF struct size attributes")

        name = extract_at_name(die)
        try:
            members = self._resolve_members(die)
        except Exception:
            self.ctx.logger.error("Error while handling %s from %s at %s %s", name, self.current_unit_path, src_file,
                                  src_line)
            raise

        if len(members) == 0:
            self.ctx.logger.debug("Struct with no members %s %s:%d", name, src_file, src_line)

        is_anon = False
        if name is None:
            # Try to see whether the struct has been typedef'ed
            try:
                typedef = self.ctx.typedef_by_offset[die.offset]
                name = typedef.name
                src_file = typedef.src_file
                src_line = typedef.src_line
            except KeyError:
                is_anon = True

        si = DWARFStructInfo(name=name,
                             size=size,
                             from_path=self.current_unit_path,
                             src_file=src_file,
                             src_line=src_line,
                             members=members,
                             is_anon=is_anon)
        self.ctx.struct_by_key[si.key] = si
        self.ctx.struct_by_offset[die.offset] = si
        if die.offset in self.ctx.struct_ref_fixup:
            for cbk in self.ctx.struct_ref_fixup[die.offset]:
                cbk(si)
            del self.ctx.struct_ref_fixup[die.offset]

    def visit_typedef(self, die):
        """
        If we find a typedef, it might be for an anonymous struct or union we are going
        to encounter later on. To resolve the correct name we store anonymous struct and union
        typedef targets for later lookup. We do not care about other typedefs as they will
        be in the type chain if the type is encountered.
        """
        target_die = die.get_DIE_from_attribute("DW_AT_type")
        if target_die.tag == "DW_TAG_structure_type" or target_die.tag == "DW_TAG_union_type":
            target_name = extract_at_name(target_die, None)
            if target_name is not None:
                # target struct is not anon, skip
                return
            typedef_name = extract_at_name(die)
            assert typedef_name is not None, "typedef without name"
            src_file, src_line = extract_at_fileline(die, self.current_debug_line)
            typedef = DWARFTypeDef(typedef_name, src_file, src_line)
            self.ctx.typedef_by_offset[target_die.offset] = typedef

            # Try to patch any anon structure we already found
            target = self.ctx.struct_by_offset.get(target_die.offset)
            if target and target.is_anon:
                target.apply_typedef(typedef)


@dataclass
class Addr2lineInfo:
    path: Path
    line: int
    symbol: str

    def __str__(self):
        return f"{self.path}:{self.line} {self.symbol}"


class Addr2lineResolver:
    def __init__(self, sdk_path: Path, obj_path: Path):
        self.addr2line_bin = sdk_path / "sdk" / "bin" / "llvm-addr2line"
        self.path = obj_path
        self.addr2line = None

    def __enter__(self):
        self.addr2line = subprocess.Popen([self.addr2line_bin, "-obj", str(self.path), "-f"],
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          text=True,
                                          encoding="utf-8")
        return self

    def map_addr(self, addr: int) -> Addr2lineInfo:
        self.addr2line.stdin.write(f"0x{addr:x}\n")
        self.addr2line.stdin.flush()
        symbol = self.addr2line.stdout.readline().strip()
        line_info = self.addr2line.stdout.readline()
        file_, line = line_info.split(":")
        return Addr2lineInfo(file_, line, symbol)

    def map_addr_range(self, start: int, end: int) -> typing.Sequence[Addr2lineInfo]:
        addr_line_info = []
        for addr in range(start, end + 1):
            self.addr2line.stdin.write(f"0x{addr:x}\n")
        self.addr2line.stdin.flush()
        for i in range(start, end + 1):
            symbol = self.addr2line.stdout.readline().strip()
            line_info = self.addr2line.stdout.readline()
            file_, line = line_info.split(":")
            addr_line_info.append(Addr2lineInfo(file_, line, symbol))
        return addr_line_info

    def __exit__(self, type_, value, traceback):
        self.addr2line.stdin.close()
        self.addr2line.terminate()
        self.addr2line.wait()


class _DWARFInfoSource:
    """
    Helper to access DWARF debug information for an ELF file.
    This currently relies on the capabilities of pyelftools and
    does not support DWARF-5.
    This is part of the main interface exposed to clients of this module.
    """
    def __init__(self, benchmark, path: Path, arch_pointer_size: int):
        self.benchmark = benchmark
        self.logger = benchmark.logger
        self.path = path
        self._fd = open(path, "rb")
        self._ef = ELFFile(self._fd)
        self._dw = self._ef.get_dwarf_info()
        self._arch_pointer_size = arch_pointer_size
        self._ctx = DWARFDataRegistry(self._ef, self._dw, self.logger, arch_pointer_size)

    def _handle_die(self, die, visitor):
        if die.tag == "DW_TAG_structure_type":
            visitor.visit_structure_type(die)
        elif die.tag == "DW_TAG_union_type":
            visitor.visit_union_type(die)
        elif die.tag == "DW_TAG_typedef":
            visitor.visit_typedef(die)

    def _handle_unit(self, unit, visitor):
        unit_die = unit.get_top_DIE()
        visitor.visit_unit(unit, unit_die)
        for die in unit_die.iter_children():
            self._handle_die(die, visitor)

    def parse_dwarf(self):
        """Main entry point to extract all the DWARF information"""
        visitors = [StructInfoVisitor(self._ctx)]
        for unit in self._dw.iter_CUs():
            # There might be an opportunity to further parallelize this if it is too slow
            for v in visitors:
                self._handle_unit(unit, v)

    def get_dwarf_data(self) -> DWARFDataRegistry:
        """Once parse_dwarf has completed, this can be used to retrieve the data registry"""
        return self._ctx

    def addr2line(self, addr_list: typing.Sequence[int]) -> typing.Sequence[typing.Tuple[Path, int]]:
        """
        Perform an addr2line lookup for now. It should be possible to do a fast lookup of the
        .debug_line information but not sure how to do it yet. TODO
        """
        with Addr2lineResolver(self.benchmark.manager_config.sdk_path, self.path) as resolver:
            lines = [resolver.map_addr(addr) for addr in addr_list]
        return lines

    def addr2line_range(self, df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
        with Addr2lineResolver(self.benchmark.manager_config.sdk_path, self.path) as resolver:
            line_start = []
            line_end = []
            for (start, end) in zip(df[start_col], df[end_col]):
                line_start.append(resolver.map_addr(start))
                line_end.append(resolver.map_addr(end))
                # lines.append(resolver.map_addr_range(start, end))
            # lines = df.apply(lambda r: resolver.map_addr_range(r[start_col], r[end_col]), axis=1)
        resolved = pd.DataFrame({"line_start": line_start, "line_end": line_end}, index=df.index)
        assert len(df) == len(resolved)
        assert df.index.equals(resolved.index)
        return resolved


################################### OLD STUFF ABOVE


class DWARFStructLayoutModel(DataModel):
    """
    Contains a dump of structure and union data layouts.

    Note that not all information is dumped here. This data model should be sufficient
    to determine padding, alignment and sub-object representablity.

    There is one row for each member, nested structures are unrolled and the
    member name is prefixed by the parent names with the C-style '.' scope operator.
    Offsets are adjusted to reflect the parent structure offset.

    Note that the type ID is not really an itentifier for our purposes. There may be
    multiple DWARF entries that refer to the same structure. This may happen if the same
    structure imported and defined in multiple compilation units.
    The index of this model should be unique, any duplicates must be removed as they
    are expected to refer to the same structure.
    The members are uniquely identified by the combination of (offset, name) which should
    be sufficient for linearized union members that share the same offset.
    """
    file: Index[str]
    line: Index[int]
    base_name: Index[str]
    member_name: Index[str]
    member_offset: Index[pd.Int64Dtype] = pa.Field(nullable=True)

    type_id: Series[int]
    size: Series[int]
    type_name: Series[str]
    member_size: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    member_type_name: Series[str] = pa.Field(nullable=True)


#: Helper to keep records  for the DWARFStructLayoutInfoModel consistent
DWARFStructLayoutRecord = namedtuple("DWARFStructLayoutRecord", [
    "type_id", "file", "line", "base_name", "member_name", "member_offset", "size", "type_name", "member_size",
    "member_type_name"
])


class StructLayoutVisitor:
    """
    Abstraction for an algorithm that visits nested members of a DWARF structure layout.

    This can be used for other algorithms that rely on the
    DWARFInfoSource.parse_struct_layout().
    """
    def __init__(self, root: pydwarf.TypeInfo):
        self.root = root

    def visit_member(self, parent: pydwarf.Member | None, curr_type: pydwarf.TypeInfo, member: pydwarf.Member,
                     prefix: str, offset: int):
        """
        Visit a leaf structure member

        :param parent: Parent member of the current type we are inspecting.
        :param curr_type: The type information of the root structure.
        :param member: The member information.
        :param prefix: The scope prefix with respect to the root structure definition.
        :param offset: The offset of the member.
        """
        pass

    def visit_empty_struct(self, parent: pydwarf.Member | None, curr_type: pydwarf.TypeInfo, prefix: str, offset: int):
        """
        Visit an empty structure. This should be used to insert markers or ignore empty structures.

        :param parent: Parent member of the current type we are inspecting.
        :param curr_type: The type information of the root structure.
        :param prefix: The scope prefix with respect to the root structure definition.
        :param offset: The offset of the empty structure member.
        """
        pass


class FlattenLayoutVisitor(StructLayoutVisitor):
    def __init__(self, benchmark, root):
        super().__init__(root)
        self.benchmark = benchmark

        #: Accumulate DWARFStructLayoutRecord tuplesrelpath
        self.records: list[DWARFStructLayoutRecord] = []

        # Normalize anonymous type names to be relative to the user src_path
        self._root_base_name = self._normalize_anon(root.base_name)
        self._root_type_name = self._normalize_anon(root.type_name)
        self._root_file = self._normalize_path(root.file)

    def _normalize_anon(self, name: str):
        if name.find("<anon>") == -1:
            return name
        match = re.match(r"(.*)<anon>\.(.*)\.([0-9]+)", name)
        if not match:
            return name
        path = Path(match.group(2))
        relpath = self._normalize_path(path)
        return match.group(1) + "<anon>." + str(relpath) + "." + match.group(3)

    def _normalize_path(self, strpath: str):
        src_root = self.benchmark.user_config.src_path
        path = Path(strpath)
        if not path.is_relative_to(src_root):
            # No changes
            return strpath
        return str(path.relative_to(src_root))

    def visit_member(self, parent, curr_type, member, prefix, offset):
        member_type_name = self._normalize_anon(member.type_info.type_name)
        record = DWARFStructLayoutRecord(type_id=self.root.handle,
                                         base_name=self._root_base_name,
                                         member_name=prefix + member.name,
                                         member_offset=offset + member.offset,
                                         file=self._root_file,
                                         line=self.root.line,
                                         size=self.root.size,
                                         type_name=self._root_type_name,
                                         member_size=member.size,
                                         member_type_name=member_type_name)
        self.records.append(record)

    def visit_empty_struct(self, parent, curr_type, prefix, offset):
        record = DWARFStructLayoutRecord(type_id=self.root.handle,
                                         base_name=self._root_base_name,
                                         member_name=prefix + "<empty>",
                                         member_offset=offset,
                                         file=self._root_file,
                                         line=self.root.line,
                                         size=self.root.size,
                                         type_name=self._root_type_name,
                                         member_size=np.nan,
                                         member_type_name=None)
        self.records.append(record)


class GraphConversionVisitor(StructLayoutVisitor):
    """
    Visitor that produces a networkx tree for a dwarf structure description

    Note that this can drop duplicate members that are identified by the tuple
    <base_name, member_name, member_offset, member_size>.
    This is done to remove multiple instances of the same structure.
    """
    #: Helper type to keep node identifiers consistent
    #: Note this should be kept in sync with :class:`DWARFStructLayoutModel`
    NodeID = namedtuple("NodeID", ["file", "line", "base_name", "member_name", "member_offset"])

    @classmethod
    def dump(cls, graph: nx.DiGraph, path: Path):
        """
        Write a GML representation of a graph.
        """
        with gzopen(path, "w") as fd:
            json.dump(graph, fd, default=nx.node_link_data)

    @classmethod
    def load(cls, path: Path) -> nx.DiGraph:
        """
        Load a GML representation of the structure layout graph
        """
        with gzopen(path, "r") as fd:
            data = json.load(fd)
        g = nx.relabel_nodes(nx.node_link_graph(data), lambda r: cls.NodeID(*r))
        g.graph["roots"] = [cls.NodeID(*r) for r in g.graph["roots"]]
        return g

    def __init__(self, graph: nx.DiGraph, benchmark, root):
        super().__init__(root)
        self.benchmark = benchmark

        self.g = graph
        self.drop_duplicates = False

        # self._type_name = fixup_anon_name(benchmark, root.base_name)
        # Normalize anonymous type names to be relative to the user src_path
        self._root_type_name = self._normalize_anon(root.base_name)
        # Normalize the file name to be relative to the user src_path
        self._root_file = self._normalize_path(root.file)

        # Create a node for the root
        self.root_id = self.NodeID(file=self._root_file,
                                   line=root.line,
                                   base_name=self._root_type_name,
                                   member_name=None,
                                   member_offset=0)
        self.g.add_node(self.root_id, type_id=root.handle, type_name=root.type_name, size=root.size)

    def _normalize_anon(self, name: str):
        if name.find("<anon>") == -1:
            return name
        match = re.match(r"(.*)<anon>\.(.*)\.([0-9]+)", name)
        if not match:
            return name
        path = Path(match.group(2))
        relpath = self._normalize_path(path)
        return match.group(1) + "<anon>." + str(relpath) + "." + match.group(3)

    def _normalize_path(self, strpath: str):
        src_root = self.benchmark.user_config.src_path
        path = Path(strpath)
        if not path.is_relative_to(src_root):
            # No changes
            return strpath
        return str(path.relative_to(src_root))

    def build_node_id(self, member: pydwarf.Member, offset: int):
        return self.NodeID(file=self._root_file,
                           line=self.root.line,
                           base_name=self._root_type_name,
                           member_name=member.name,
                           member_offset=offset + member.offset)

    def visit_member(self, parent, curr_type, member, prefix, offset):
        if parent is None:
            parent_id = self.root_id
        else:
            parent_id = self.build_node_id(parent, offset - parent.offset)
        node_id = self.build_node_id(member, offset)
        if node_id in self.g.nodes:
            # Nothing to do, maybe safety-check for equality
            return

        attrs = {
            "type_id": self.root.handle,
            "type_name": self._normalize_anon(member.type_info.type_name),
            "size": member.size
        }
        self.g.add_node(node_id, **attrs)
        self.g.add_edge(parent_id, node_id)

    def visit_empty_struct(self, parent, curr_type, prefix, offset):
        return


class DWARFInfoSource:
    """
    Wrapper around the LLVM native dwarf extraction library.
    """
    def __init__(self, benchmark, path: Path, arch_pointer_size: int):
        self.benchmark = benchmark
        self.logger = benchmark.logger
        self.path = path
        self._helper = pydwarf.DWARFHelper(str(path))
        self._arch_pointer_size = arch_pointer_size

    def _do_visit_layout(self, visitor: StructLayoutVisitor, parent_members: list[pydwarf.Member],
                         curr: pydwarf.TypeInfo, prefix: str, offset: int):
        """
        Visit nested members by linearizing the structure layout.

        :param visitor: The visitor strategy.
        :param parent_members: A stack-like list of parent members we are recursing into.
        :param curr: The type information we have to visit at this stage.
        :param prefix: Name prefix from the parent members.
        :param offset: Offset of the current member we have to visit.
        """
        records = []
        parent = parent_members[-1] if parent_members else None

        for member in curr.layout:
            visitor.visit_member(parent, curr, member, prefix, offset)
            if (member.type_info.flags & (pydwarf.TypeInfoFlags.kIsStruct | pydwarf.TypeInfoFlags.kIsUnion)):
                # Recurse into the structure/union
                parent_members.append(member)
                self._do_visit_layout(visitor, parent_members, member.type_info, prefix + f"{member.name}.",
                                      offset + member.offset)
                parent_members.pop()
        if len(curr.layout) == 0:
            # Special case for empty structures / unions.
            # Always emit one entry with missing member name
            visitor.visit_empty_struct(parent, curr, prefix, offset)

    def visit_layout(self, visitor: StructLayoutVisitor):
        self._do_visit_layout(visitor, [], visitor.root, "", 0)

    def load_type_info(self) -> pydwarf.TypeInfoContainer:
        return self._helper.collect_type_info()

    def build_struct_layout_graph(self,
                                  container: pydwarf.TypeInfoContainer,
                                  g: nx.DiGraph | None = None) -> nx.DiGraph:
        """
        Given a TypeInfoContainer collection, produce a networkx graph that represents its
        structure.

        This should allow better flexibility with the algorithms we run on the graph.
        XXX I wonder whether I can do this in C++ directly...
        """
        if g is None:
            g = nx.DiGraph()
        if "roots" not in g.graph:
            g.graph["roots"] = []
        roots = []

        for type_info in container.iter_composite():
            v = GraphConversionVisitor(g, self.benchmark, type_info)
            self.visit_layout(v)
            roots.append(v.root_id)

        g.graph["roots"] += roots
        return g

    def parse_struct_layout(self,
                            container: pydwarf.TypeInfoContainer,
                            root: pydwarf.TypeInfo | None = None) -> DataFrame[DWARFStructLayoutModel]:
        """
        Recursively dump the structure layout. Optionally, this can be restricted to a single TypeInfo node
        instead of parsing everything.

        XXX rename this as flatten_struct_layout and use the graph as input.
        """
        records = []
        if root:
            v = FlattenLayoutVisitor(self.benchmark, root)
            self.visit_layout(v)
            records = v.records
        else:
            for type_info in container.iter_composite():
                v = FlattenLayoutVisitor(self.benchmark, type_info)
                self.visit_layout(v)
                records += v.records
        df = pd.DataFrame.from_records(records, columns=DWARFStructLayoutRecord._fields)
        df.set_index(["file", "line", "base_name", "member_name", "member_offset"], inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        return df


class DWARFManager:
    """
    Manages DWARF information for all object files imported by a
    benchmark instance.
    Each source object file is registered using a unique name
    (akin to what is done with the symbolizer). At some point
    we may need to integrate this with the symbolizer in some way.
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.logger = new_logger(f"{benchmark.uuid}.DWARFHelper")
        self.benchmark = benchmark
        self.objects = {}

    def register_object(self, obj_key: str, path: Path, arch_pointer_size=8) -> DWARFInfoSource:
        if obj_key in self.objects:
            # Check that the path also matches
            if self.objects[obj_key].path == path:
                return
        assert obj_key not in self.objects, "Duplicate DWARF object source"
        self.objects[obj_key] = DWARFInfoSource(self.benchmark, path, arch_pointer_size=arch_pointer_size)
        return self.objects[obj_key]

    def get_object(self, obj_key: str) -> typing.Optional[DWARFInfoSource]:
        return self.objects.get(obj_key, None)

    def addr2line_range_to_df(self, df: pd.DataFrame, addr_start_column: str, addr_end_column: str, as_key_column: str):
        """
        Create a new dataframe with the same index as the given one, populated with the line numbers
        associated to the addresses in the given column.
        """
        assert False, "Disabled, must reimplement"

        assert as_key_column not in df.index.names
        assert as_key_column in df.columns
        resolved = []
        grouped = df.groupby(as_key_column)
        for as_key, chunk in grouped:
            self.logger.debug("Resolving line numbers for %s: %d ranges", as_key, len(chunk))
            info_source = self.get_object(as_key)
            if not info_source:
                self.logger.debug("DWARF data for %s not found, use kernel", as_key)
                info_source = self.get_object("kernel.full")
            if not info_source:
                self.logger.warning("Missing DWARF data for %s", as_key)
                continue
            line_info = info_source.addr2line_range(chunk, addr_start_column, addr_end_column)
            assert line_info.index.equals(chunk.index)
            resolved.append(line_info)
        out_df = pd.concat(resolved, axis=0).sort_index()
        assert out_df.index.equals(df.index)
        return out_df
