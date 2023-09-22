import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, HTMLTemplateTarget, LocalFileTarget, Target)
from ..core.config import Config, ConfigPath, InstanceKernelABI
from ..core.dwarf import DWARFManager, GraphConversionVisitor
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import BenchmarkTask, DataGenTask, dependency, output
from ..ext import pychericap, pydwarf
from .model import (ImpreciseSubobjectInfoModel, ImpreciseSubobjectInfoModelRecord, ImpreciseSubobjectLayoutModel)


@dataclass
class PathMatchSpec(Config):
    path: ConfigPath
    match: str | None = None


@dataclass
class ExtractImpreciseSubobjectConfig(Config):
    """
    Configure the imprecise sub-object extractor.
    """
    #: List of paths that are used to extract DWARF information.
    #: Note that multiple paths will be considered as belonging to the same "object",
    #: if multiple objects are being compared, we should use parameterization.
    #: Note that relative paths will be interpreted as relative paths into a cheribuild rootfs.
    dwarf_data_sources: list[PathMatchSpec]


class ValueTarget(Target):
    def __init__(self, task, output_id: str | None = None):
        self._value = None
        # Borg state initialization occurs here
        super().__init__(task, output_id)

    def assign(self, v: any):
        self._value = v

    def get(self) -> any:
        return self._value


class StructLayoutLoader(BenchmarkTask):
    """
    Custom task to load the structure layout graph from GML
    """
    task_namespace = "subobject"
    task_name = "imprecise-layout-loader"

    def __init__(self, benchmark, target):
        assert target.is_file()
        self.target = target
        # Borg state initialization occurs here
        super().__init__(benchmark)

    @output
    def graph(self):
        return ValueTarget(self)

    def run(self):
        g = GraphConversionVisitor.load(self.target.path)
        self.graph.assign(g)


class ExtractImpreciseSubobject(DataGenTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.

    This will produce a graph containing structure layouts as trees.
    XXX Document tree structure and node attributes.
    """
    public = True
    task_namespace = "subobject"
    task_name = "extract-imprecise"
    task_config_class = ExtractImpreciseSubobjectConfig

    def __init__(self, benchmark, script, task_config):
        super().__init__(benchmark, script, task_config)

        instance_config = self.benchmark.config.instance
        if instance_config.cheri_target.is_riscv():
            # XXX support/detect riscv32
            self.cap_format = pychericap.CompressedCap128
        elif instance_config.cheri_target.is_morello():
            self.cap_format = pychericap.CompressedCap128m
        else:
            self.logger.error("DWARF TypeInfo extraction unsupported for %s", instance_config.cheri_target)
            raise RuntimeError(f"Unsupported instance target {instance_config.cheri_target}")

    @output
    def imprecise_layouts(self):
        return LocalFileTarget(self, ext="gml.gz", loader=StructLayoutLoader)

    def _check_imprecise(self, offset: int, size: int) -> tuple[int, int] | None:
        """
        Check if a specific member of a structure is representable.

        If it is not representable, return a tuple (new_base, new_top)
        """
        subobject_cap = self.cap_format.make_max_bounds_cap(offset)
        is_exact = subobject_cap.setbounds(size)

        if is_exact:
            assert subobject_cap.base() == offset and subobject_cap.top() == offset + size
            return None
        assert subobject_cap.base() < offset or subobject_cap.top() >= offset + size
        return (subobject_cap.base(), subobject_cap.top())

    def _find_imprecise_for(self, g: nx.DiGraph, n: "NodeID"):
        """
        """
        g.nodes[n]["has_imprecise"] = False
        alias_group_id = 0
        imprecise = set()
        for parent, child in nx.dfs_edges(g, source=n):
            # Determine if this node is imprecise
            result = self._check_imprecise(child.member_offset, g.nodes[child]["size"])
            if result:
                g.nodes[child]["alias_group_id"] = alias_group_id
                alias_group_id += 1
                g.nodes[child]["alias_aligned_base"] = result[0]
                g.nodes[child]["alias_aligned_top"] = result[1]
                imprecise.add(child)
                g.nodes[n]["has_imprecise"] = True

        if len(imprecise) == 0:
            # Bail, nothing else to do
            return

        imprecise_table = pd.Series(map(lambda i: g.nodes[i]["alias_group_id"], imprecise),
                                    index=pd.IntervalIndex.from_arrays(map(lambda i: g.nodes[i]["alias_aligned_base"],
                                                                           imprecise),
                                                                       map(lambda i: g.nodes[i]["alias_aligned_top"],
                                                                           imprecise),
                                                                       closed="left"))
        # Now need to determine the alias groups
        for parent, child in nx.dfs_edges(g, source=n):
            # XXX reject unions?
            aliasing = imprecise_table.index.overlaps(
                pd.Interval(child.member_offset, child.member_offset + g.nodes[child]["size"], closed="left"))
            if not aliasing.any():
                continue
            groups = set(imprecise_table[aliasing].unique())
            # Ignore aliasing groups of descendants and ancestors of this node.
            # This represents the fact that the subobject for a whole structure does
            # not really alias any of its members.
            remove_groups = set()
            remove_groups.add(g.nodes[child].get("alias_group_id", None))
            for d in nx.descendants(g, child):
                remove_groups.add(g.nodes[d].get("alias_group_id", None))
            for d in nx.ancestors(g, child):
                remove_groups.add(g.nodes[d].get("alias_group_id", None))
            groups = groups.difference(remove_groups)
            if groups:
                # Can not use set because it is harder to serialize to GML cleanly
                g.nodes[child]["alias_groups"] = list(map(int, groups))
        self.logger.debug("Found %d imprecise members for %s", len(imprecise), n)

    def _find_imprecise(self, g: nx.DiGraph):
        """
        Generate alias groups for imprecise sub-object members.
        This generates a graph representing data structure layouts in the dwarf information.
        The alias_group_id attribute contains an integer identifier for the set of fields aliasing
        with the one with the attribute.
        The alias_aligned_base and alias_aligned_top contain the base and top offsets after CHERI
        representability rounding.
        The alias_groups attribute contains a list of all the group IDs that a field is aliasing with.
        """
        roots = g.graph["roots"]
        if len(roots) == 0:
            self.logger.debug("No data structures for %s", dw.path)
            return None
        for n in roots:
            self._find_imprecise_for(g, n)
        return g

    def _resolve_paths(self, target: PathMatchSpec) -> Iterator[Path]:
        """
        Resolve a path matcher to a list of paths.
        """
        if target.match is None:
            yield from [target.path]
        matcher = re.compile(target.match)

        def match_path(p: Path):
            r = p.relative_to(target.path)
            return matcher.match(str(r))

        yield from filter(match_path, target.path.rglob("*"))

    def run(self):
        iconf = self.benchmark.config.instance
        layout_graph = nx.DiGraph()

        for target in self.config.dwarf_data_sources:
            for item in self._resolve_paths(target):
                self.logger.debug("Inspect subobject bounds from %s", item)
                if not item.is_file():
                    self.logger.error("File %s is not a regular file, skipping", item)
                    continue
                # XXX may want to make the manager use a more predictable target name
                # XXX the pointer size detection should go away
                dw = self.benchmark.dwarf.register_object(item, item, arch_pointer_size=iconf.user_pointer_size)
                info = dw.load_type_info()
                dw.build_struct_layout_graph(info, layout_graph)

        self._find_imprecise(layout_graph)
        layout_graph.graph["dataset_id"] = str(self.benchmark.uuid)
        layout_graph.graph["dataset_gid"] = str(self.benchmark.g_uuid)
        # Now we can dump the layout
        GraphConversionVisitor.dump(layout_graph, self.imprecise_layouts.path)


class ImpreciseSubobjectBoundsUnion(AnalysisTask):
    """
    Merge all imprecise subobject bounds warnings.

    This merges the datasets by platform g_uuid, so it is possible
    to observe the difference between the behaviour for Morello and
    RISC-V variants.
    """
    task_namespace = "subobject"
    task_name = "imprecise-subobject-union"

    @dependency
    def layout_data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(ExtractImpreciseSubobject)
            yield task.imprecise_layouts.get_loader()

    def run(self):
        # Note that we want to remove duplicates that show the same structure label across
        # different files if they have the same type, offset and size
        data = [d.graph.get() for d in self.layout_data]
        self.all_layouts.assign(data)

    @output
    def all_layouts(self):
        return ValueTarget(self)


class ImpreciseMembersPlotBase(PlotTask):
    """
    Produce a plot showing the difference in size and alignment caused by CHERI
    representability rounding on each sub-object.

    Each different platform (gid) is rendered separately.
    """
    task_namespace = "subobject"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    def _collect_imprecise_members(self, layouts: list[nx.DiGraph]):
        """
        Construct a dataframe containing all the imprecise structure members
        """
        imprecise = []
        for g in layouts:
            gid = uuid.UUID(g.graph["dataset_gid"])
            records = []
            for struct_root in g.graph["roots"]:
                if not g.nodes[struct_root].get("has_imprecise", False):
                    continue
                for desc in nx.descendants(g, struct_root):
                    if "alias_group_id" not in g.nodes[desc]:
                        continue
                    # This is an imprecise member, remember it
                    r = ImpreciseSubobjectInfoModelRecord(file=desc.file,
                                                          line=desc.line,
                                                          base_name=desc.base_name,
                                                          member_name=desc.member_name,
                                                          member_offset=desc.member_offset,
                                                          member_size=g.nodes[desc]["size"],
                                                          member_aligned_base=g.nodes[desc]["alias_aligned_base"],
                                                          member_aligned_top=g.nodes[desc]["alias_aligned_top"])
                    records.append(r)
            df = pd.DataFrame.from_records(records, columns=ImpreciseSubobjectInfoModelRecord._fields)
            if len(df) == 0:
                continue
            df["dataset_gid"] = gid
            imprecise.append(df)
        if imprecise:
            df = pd.concat(imprecise, axis=0)
        else:
            df = pd.DataFrame([], columns=("dataset_gid", ) + ImpreciseSubobjectInfoModelRecord._fields)
        df = df.set_index(["dataset_gid", "file", "line", "base_name", "member_name", "member_offset"])
        return df[~df.index.duplicated(keep="first")]

    def _prepare_dataset(self):
        """
        Prepare the imprecise subobjects dataset from the loaded graph layouts
        """
        df = self._collect_imprecise_members(self.data.all_layouts.get())
        # Normalize base and size with respect to the "requested" base offset
        # and size
        member_offset = df.index.get_level_values("member_offset")
        df["aligned_size"] = df["member_aligned_top"] - df["member_aligned_base"]
        df["base_rounding"] = member_offset - df["member_aligned_base"]
        df["top_rounding"] = df["member_aligned_top"] - (member_offset + df["member_size"])

        assert (df["base_rounding"] >= 0).all()
        assert (df["top_rounding"] >= 0).all()
        return df

    def _prepare_dataset_filter_common(self):
        """
        Same as _prepare_dataset(), but only retain structures that are built
        for all dataset_gid (E.g. common to both morello and riscv).
        """
        df = self._prepare_dataset()
        layouts = self.data.all_layouts.get()

        # Filter the dataframe by structures that exist in all layout dumps
        # To do so, find the intersection between all structure identifiers, then
        # select only the nodes in df that belong to these structures.
        # This means joining on [file, line, base_name]
        common = set(layouts[0].graph["roots"])
        for g in layouts[1:]:
            common = common.intersection(g.graph["roots"])
        common_df = pd.DataFrame.from_records(map(lambda n: (n.file, n.line, n.base_name), common),
                                              columns=["file", "line", "base_name"])
        return df.join(common_df.set_index(["file", "line", "base_name"]), how="inner")

    def _legend_on_top(self, fig, ax, **kwargs):
        """
        Fixup the legend to appear at the top of the axes
        """
        if not fig.legends:
            return
        # Hack to remove the legend as we can not easily move it
        # Not sure why seaborn puts the legend in the figure here
        legend = fig.legends.pop()
        ax.legend(legend.legend_handles,
                  map(lambda t: t.get_text(), legend.texts),
                  loc='center',
                  bbox_to_anchor=(0, 1.02),
                  ncols=4,
                  **kwargs)


class AllImpreciseMembersPlot(ImpreciseMembersPlotBase):
    """
    Variant of imprecise subobject bounds plot that shows the amount of base and top
    aliasing for each imprecise structure member.
    """
    public = True
    task_name = "imprecise-subobject-plot"

    def _plot_imprecise(self, df):
        """
        Produce the imprecise fields vs base and top rounding.
        """
        show_df = df.reset_index().melt(id_vars=df.index.names + ["member_size"],
                                        value_vars=["base_rounding", "top_rounding"],
                                        var_name="source_of_imprecision")
        show_df["legend"] = (show_df["source_of_imprecision"] + " " + show_df["dataset_gid"].map(self.g_uuid_to_label))
        show_df["label"] = show_df["base_name"] + "::" + show_df["member_name"]

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(
                so.Bar(),
                so.Dodge(by=["dataset_gid", "source_of_imprecision"]),
                # so.Stack(),
                dataset_gid=show_df["dataset_gid"],
                source_of_imprecision=show_df["source_of_imprecision"],
                orient="y").plot())

            self._legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    def _plot_size(self, df):
        """
        Plot the imprecise_field vs size.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "dataset_gid"])

        # Basic field vs size plot (horizontal Y)
        with new_figure(self.imprecise_fields_size_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()

            (so.Plot(show_df, y="label", x="member_size", color="legend").on(ax).add(so.Bar(),
                                                                                     so.Dodge(by=["dataset_gid"]),
                                                                                     dataset_gid=show_df["dataset_gid"],
                                                                                     orient="y").plot())

            self._legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Sub-object size (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

        # Distribution of sizes as an histogram, without field names
        with new_figure(self.imprecise_fields_size_hist.paths()) as fig:
            ax = fig.subplots()

            min_size = int(np.log2(show_df["member_size"].min()))
            max_size = int(np.log2(show_df["member_size"].max())) + 1
            hist_bins = [2**x for x in range(min_size, max_size)]

            (so.Plot(show_df, "member_size", color="label").add(so.Area(), so.Hist(bins=hist_bins)).plot())
            self._legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Sub-object size (Bytes)")
            ax.set_ylabel("# of imprecise sub-objects")

    def run_plot(self):
        df = self._prepare_dataset()
        sns.set_theme()

        self._plot_imprecise(df)
        self._plot_size(df)

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self, prefix="imprecision")

    @output
    def imprecise_fields_size_plot(self):
        return PlotTarget(self, prefix="size")

    @output
    def imprecise_fields_size_hist(self):
        return PlotTarget(self, prefix="size-hist")


class ImpreciseCommonMembersPlot(ImpreciseMembersPlotBase):
    """
    This is similar to :class:`AllImpreciseMembersPlot` but only displays imprecise struct
    members for structures that are present for all dataset_id values.
    This means that structures that are only compiled for Morello or RISC-V are not displayed.
    """
    public = True
    task_name = "imprecise-common-subobject-plot"

    def run_plot(self):
        df = self._prepare_dataset_filter_common()

        sns.set_theme()

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50), constrained_layout=False) as fig:
            ax = fig.subplots()
            show_df = df.reset_index().melt(id_vars=df.index.names,
                                            value_vars=["base_rounding", "top_rounding"],
                                            var_name="source_of_imprecision")
            show_df["legend"] = (show_df["source_of_imprecision"] + " " +
                                 show_df["dataset_gid"].map(self.g_uuid_to_label))
            show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(so.Bar(),
                                                                               so.Dodge(by=["dataset_gid"]),
                                                                               so.Stack(),
                                                                               dataset_gid=show_df["dataset_gid"],
                                                                               orient="y").plot())

            self._legend_on_top(fig, ax)
            # ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self)


class RequiredSubobjectPrecision(ImpreciseMembersPlotBase):
    """
    Plot the number of bits required to ensure that imprecise sub-objects would be
    representable.
    """
    public = True
    task_name = "imprecise-subobject-bits-plot"

    @output
    def imprecise_bits_plot(self):
        return PlotTarget(self, prefix="all")

    @output
    def imprecise_bits_cdf(self):
        return PlotTarget(self, prefix="cdf")

    @output
    def imprecise_common_bits_plot(self):
        return PlotTarget(self, prefix="common")

    @output
    def imprecise_common_bits_cdf(self):
        return PlotTarget(self, prefix="common-cdf")

    def _compute_precision(self, base, top):
        """
        This should produce the mantissa size required for a specific (base, top) pair
        to be representable.
        """
        assert base <= top, "Invalid base > top"

        def lsb(x):
            if x == 0:
                return 0
            return int(np.log2(x & -x))

        len_msb = np.floor(np.log2(top - base)) if top != base else 0
        if top == 0:
            # Base must also be 0
            exponent = 0
        elif base == 0:
            exponent = lsb(top)
        else:
            exponent = min(lsb(base), lsb(top))
        return len_msb - exponent + 1

    def _compute_platform_precision(self, g_uuid, base, top):
        config = self.get_instance_config(g_uuid)
        if config.cheri_target.is_riscv():
            mantissa_width = pychericap.CompressedCap128.get_mantissa_width()
            exponent_width = 3
        elif config.cheri_target.is_morello():
            mantissa_width = pychericap.CompressedCap128m.get_mantissa_width()
            exponent_width = 3
        else:
            self.logger.error("Unsupported cheri_target=%s", config.cheri_target)
            raise RuntimeError(f"Unsupported instance target {config.cheri_target}, "
                               "need to implement")
        ie_threshold = mantissa_width - 2
        len_msb = np.floor(np.log2(top - base)) if top != base else 0
        if len_msb + 1 <= ie_threshold:
            precision = ie_threshold
        else:
            precision = mantissa_width - (1 + exponent_width)
        return precision

    def _plot_precision_bars(self, df, target):
        """
        Produce a bar plot showing the extra number of precision bits required
        for each imprecise sub-object member.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "label", "dataset_gid"])

        with new_figure(target.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()
            (so.Plot(show_df, y="label", x="additional_precision",
                     color="legend").on(ax).add(so.Bar(),
                                                so.Dodge(by=["dataset_gid"]),
                                                dataset_gid=show_df["dataset_gid"],
                                                orient="y").plot())

            self._legend_on_top(fig, ax)
            ax.set_xlabel("Increased precision (bits) required")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    def _plot_precision_cdf(self, df, target):
        """
        Produce a CDF plot showing the amount of imprecise sub-object members that
        can be "fixed" by adding a number of precision bits.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "label", "dataset_gid"])

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()

            sns.ecdfplot(data=show_df, x="additional_precision", hue="legend", ax=ax)
            ax.set_xlabel("Additional precision bits")
            ax.set_ylabel("Proportion of sub-objects that become representable")

    def _plot_precision(self, df, bars_target, cdf_target):
        sns.set_theme()
        # Compute imprecision bits
        # XXX just validate the model here?
        member_offset_index = df.index.names.index("member_offset")
        assert member_offset_index >= 0, "Missing member_offset from index"
        gid_index = df.index.names.index("dataset_gid")
        assert gid_index >= 0, "Missing dataset_gid from index"

        def _calc_precision(r):
            return self._compute_precision(r.name[member_offset_index], r.name[member_offset_index] + r["member_size"])

        df["required_precision"] = df.apply(_calc_precision, axis=1)

        def _calc_plat_precision(r):
            return self._compute_platform_precision(r.name[gid_index], r.name[member_offset_index],
                                                    r.name[member_offset_index] + r["member_size"])

        df["platform_precision"] = df.apply(_calc_plat_precision, axis=1)

        df["additional_precision"] = df["required_precision"] - df["platform_precision"]
        assert (df["additional_precision"] >= 0).all(), \
            "Something is wrong, these must be unrepresentable"

        self._plot_precision_bars(df, bars_target)
        self._plot_precision_cdf(df, cdf_target)

    def run_plot(self):
        self._plot_precision(self._prepare_dataset(), self.imprecise_bits_plot, self.imprecise_bits_cdf)
        self._plot_precision(self._prepare_dataset_filter_common(), self.imprecise_common_bits_plot,
                             self.imprecise_common_bits_cdf)


class ImpreciseSetboundsLayouts(PlotTask):
    """
    Produce a D3-js html document that allows to browse the structure layouts with imprecision
    and inspect the data members that are affected by imprecision.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-layouts"

    class LayoutDescription:
        ID_COUNTER = 0

        @classmethod
        def next_id(cls):
            cls.ID_COUNTER += 1
            return cls.ID_COUNTER

        def __init__(self, graph, root):
            self._graph = graph
            self._root = root
            self.id = f"layout-{self.next_id()}"

        @property
        def name(self):
            return self._root.base_name

        @property
        def member_name(self):
            return self._root.member_name

        @property
        def member_offset(self):
            return self._root.member_offset

        @property
        def member_type(self):
            return self._graph.nodes[self._root]["type_name"]

        @property
        def member_size(self):
            return self._graph.nodes[self._root]["size"]

        @property
        def is_imprecise(self):
            return self.alias_id is not None

        @property
        def alias_id(self):
            return self._graph.nodes[self._root].get("alias_group_id", None)

        @property
        def alias_groups(self):
            return ",".join(map(str, self._graph.nodes[self._root].get("alias_groups", [])))

        @property
        def location(self):
            return f"{self._root.file}:{self._root.line}"

        @property
        def has_children(self):
            return self._graph.out_degree(self._root) > 0

        def __iter__(self):
            for n in self._graph.successors(self._root):
                yield self.__class__(self._graph, n)

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    @output
    def layouts_table(self):
        return HTMLTemplateTarget(self, "imprecise-subobject-layout.html.jinja")

    def run(self):
        layouts = self.data.all_layouts.get()

        layout_groups = {}
        for g in layouts:
            group_name = self.g_uuid_to_label(g.graph["dataset_gid"])
            descriptions = []
            self.logger.debug("Prepare structures for %s", group_name)
            for struct_root in set(g.graph["roots"]):
                if not g.nodes[struct_root].get("has_imprecise", False):
                    continue
                descriptions.append(self.LayoutDescription(g, struct_root))
            layout_groups[group_name] = descriptions

        self.layouts_table.render(layout_groups=layout_groups)
