import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import Config, ConfigPath, DatasetArtefact, DatasetName
from ..core.csv import CSVDataSetContainer
from ..core.dataset import Field
from ..core.procstat import ProcstatDataset


@dataclass
class NetperfRunConfig(Config):
    #: Path to netperf/netserver in the guest
    netperf_path: ConfigPath = Path("opt/{instance.cheri_target}/netperf/bin")

    #: Actual benchmark options
    netperf_options: typing.List[str] = field(default_factory=list)

    #: Netserver options (used for both priming and the actual benchmark)
    netserver_options: typing.List[str] = field(default_factory=list)

    #: use KTRACE netserver to resolve forked netserver PIDs?
    netserver_resolve_forks: bool = False


class NetperfProcstat(ProcstatDataset):
    """
    Specialized netperf procstat dataset generator/parser.
    """
    dataset_config_name = DatasetName.PROCSTAT_NETPERF

    def gen_pre_benchmark(self, script):
        netperf = self.benchmark.get_dataset(DatasetName.NETPERF_DATA)
        assert netperf, "Netperf dataset is missing"
        netserver = script.gen_bg_cmd(netperf.netserver_bin, ["-p", "9998", "-D"], env=netperf.run_env)
        netperf_stopped = script.gen_bg_cmd(netperf.netperf_bin, ["-z"], env=netperf.run_env)
        # Sleep to give some time to netperf to settle in the background
        script.gen_sleep(5)
        self._gen_run_procstat(script, netperf_stopped)
        self._gen_run_procstat(script, netserver, header=False)
        script.gen_stop_bg_cmd(netperf_stopped)
        script.gen_stop_bg_cmd(netserver)


class NetperfData(CSVDataSetContainer):
    dataset_config_name = DatasetName.NETPERF_DATA
    dataset_source_id = DatasetArtefact.NETPERF
    run_options_class = NetperfRunConfig
    fields = [
        Field.str_field("Socket Type"),
        Field.str_field("Protocol"),
        Field.str_field("Direction"),
        Field.data_field("Elapsed Time (sec)"),
        Field.data_field("Throughput"),
        Field.str_field("Throughput Units"),
        Field("Local Send Socket Size Requested"),
        Field("Local Send Socket Size Initial"),
        Field("Local Send Socket Size Final"),
        Field("Local Recv Socket Size Requested"),
        Field("Local Recv Socket Size Initial"),
        Field("Local Recv Socket Size Final"),
        Field("Remote Send Socket Size Requested"),
        Field("Remote Send Socket Size Initial"),
        Field("Remote Send Socket Size Final"),
        Field("Remote Recv Socket Size Requested"),
        Field("Remote Recv Socket Size Initial"),
        Field("Remote Recv Socket Size Final"),
        Field("Local Send Size"),
        Field("Local Recv Size"),
        Field("Remote Send Size"),
        Field("Remote Recv Size"),
        Field.index_field("Request Size Bytes"),
        Field.index_field("Response Size Bytes"),
        Field("Local CPU Util %"),
        Field("Local CPU User %"),
        Field("Local CPU System %"),
        Field("Local CPU I/O %"),
        Field("Local CPU IRQ %"),
        Field("Local CPU swintr %"),
        Field.str_field("Local CPU Util Method"),
        Field("Local Service Demand"),
        Field("Remote CPU Util %"),
        Field("Remote CPU User %"),
        Field("Remote CPU System %"),
        Field("Remote CPU I/O %"),
        Field("Remote CPU IRQ %"),
        Field("Remote CPU swintr %"),
        Field.str_field("Remote CPU Util Method"),
        Field("Remote Service Demand"),
        Field.str_field("Service Demand Units"),
        Field("Confidence Level Percent"),
        Field("Confidence Width Target"),
        Field("Confidence Iterations Run"),
        Field("Throughput Confidence Width (%)"),
        Field("Local CPU Confidence Width (%)"),
        Field("Remote CPU Confidence Width (%)"),
        Field.data_field("Transaction Rate Tran/s"),
        Field.data_field("Round Trip Latency usec/tran"),
        Field("Initial Burst Requests"),
        Field("Local Transport Retransmissions"),
        Field("Remote Transport Retransmissions"),
        Field("Transport MSS bytes"),
        Field("Local Send Throughput"),
        Field("Local Recv Throughput"),
        Field("Remote Send Throughput"),
        Field("Remote Recv Throughput"),
        Field("Local CPU Bind"),
        Field("Local CPU Count"),
        Field("Local Peak Per CPU Util %"),
        Field("Local Peak Per CPU ID"),
        Field("Local CPU Frequency MHz"),
        Field("Remote CPU Bind"),
        Field("Remote CPU Count"),
        Field("Remote Peak Per CPU Util %"),
        Field("Remote Peak Per CPU ID"),
        Field("Remote CPU Frequency MHz"),
        Field("Source Port"),
        Field.str_field("Source Address"),
        Field("Source Family"),
        Field("Destination Port"),
        Field.str_field("Destination Address"),
        Field("Destination Family"),
        Field("Local Send Calls"),
        Field("Local Recv Calls"),
        Field("Local Bytes Per Recv"),
        Field("Local Bytes Per Send"),
        Field("Local Bytes Sent"),
        Field("Local Bytes Received"),
        Field("Local Bytes Xferred"),
        Field("Local Send Offset"),
        Field("Local Recv Offset"),
        Field("Local Send Alignment"),
        Field("Local Recv Alignment"),
        Field("Local Send Width"),
        Field("Local Recv Width"),
        Field("Local Send Dirty Count"),
        Field("Local Recv Dirty Count"),
        Field("Local Recv Clean Count"),
        Field("Local NODELAY"),
        Field("Local Cork"),
        Field("Remote Send Calls"),
        Field("Remote Recv Calls"),
        Field("Remote Bytes Per Recv"),
        Field("Remote Bytes Per Send"),
        Field("Remote Bytes Sent"),
        Field("Remote Bytes Received"),
        Field("Remote Bytes Xferred"),
        Field("Remote Send Offset"),
        Field("Remote Recv Offset"),
        Field("Remote Send Alignment"),
        Field("Remote Recv Alignment"),
        Field("Remote Send Width"),
        Field("Remote Recv Width"),
        Field("Remote Send Dirty Count"),
        Field("Remote Recv Dirty Count"),
        Field("Remote Recv Clean Count"),
        Field("Remote NODELAY"),
        Field("Remote Cork"),
        Field.str_field("Local Interface Vendor"),
        Field.str_field("Local Interface Device"),
        Field.str_field("Local Interface Subvendor"),
        Field.str_field("Local Interface Subdevice"),
        Field.str_field("Remote Interface Vendor"),
        Field.str_field("Remote Interface Device"),
        Field.str_field("Remote Interface Subvendor"),
        Field.str_field("Remote Interface Subdevice"),
        Field("Local Interval Usecs"),
        Field("Local Interval Burst"),
        Field("Remote Interval Usecs"),
        Field("Remote Interval Burst"),
        Field("Local OS Security Type ID"),
        Field("Local OS Security Enabled Num"),
        Field("Result Tag"),
        Field.str_field("Test UUID"),
        Field("Minimum Latency Microseconds"),
        Field("Maximum Latency Microseconds"),
        Field("50th Percentile Latency Microseconds"),
        Field("90th Percentile Latency Microseconds"),
        Field("99th Percentile Latency Microseconds"),
        Field("Mean Latency Microseconds"),
        Field("Stddev Latency Microseconds"),
        Field("Local Socket Priority"),
        Field("Remote Socket Priority"),
        Field.str_field("Local Socket TOS"),
        Field.str_field("Remote Socket TOS"),
        Field("Local Congestion Control Algorithm"),
        Field("Remote Congestion Control Algorithm"),
        Field("Local Fill File"),
        Field("Remote Fill File"),
        Field.str_field("Command Line"),
        Field.str_field("CHERI Netperf ABI"),
        Field.str_field("CHERI Kernel ABI")
    ]

    def __init__(self, benchmark, config):
        super().__init__(benchmark, config)
        self.netserver_bin = None
        self.netperf_bin = None
        self.run_env = {"STATCOUNTERS_NO_AUTOSAMPLE": "1"}
        self.netserver_task = None

    @property
    def has_qemu(self):
        if (self.benchmark.get_dataset(DatasetName.QEMU_STATS_BB_HIT) is not None
                or self.benchmark.get_dataset(DatasetName.QEMU_STATS_BB_ICOUNT) is not None
                or self.benchmark.get_dataset(DatasetName.QEMU_STATS_CALL_HIT) is not None
                or self.benchmark.get_dataset(DatasetName.QEMU_UMA_COUNTERS) is not None):
            return True
        return False

    @property
    def has_pmc(self):
        if self.benchmark.get_dataset(DatasetName.PMC) is not None:
            return True
        return False

    @property
    def has_pmc_profclock(self):
        if self.benchmark.get_dataset(DatasetName.PMC_PROFCLOCK_STACKSAMPLE) is not None:
            return True
        return False

    def _load_csv(self, path: Path, **kwargs):
        kwargs["skiprows"] = 1
        return super()._load_csv(path, **kwargs)

    def _kdump_output_path(self):
        return self.benchmark.get_benchmark_data_path() / f"netserver-ktrace-{self.benchmark.uuid}.txt"

    def load(self):
        pidmap = self.benchmark.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        if pidmap and self.config.netserver_resolve_forks:
            kdump_out = self._kdump_output_path()
            # Load the kdump auxiliary data to resolve extra PIDs
            self.logger.info("Loading netserver PIDs from auxiliary kdump %s", kdump_out)
            with open(kdump_out, "r") as kdump_fd:
                pidmap.load_from_kdump(kdump_fd)

    def load_iteration(self, iteration):
        path = self.iteration_output_file(iteration)
        csv_df = self._load_csv(path)
        csv_df["iteration"] = iteration
        self._append_df(csv_df)

    def aggregate(self):
        super().aggregate()
        self.agg_df = self.merged_df.copy()

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    def get_addrspace_key(self):
        return self.netperf_bin.name

    def _set_netperf_option(self, flag, value, replace=False):
        """
        Set a netperf CLI option,  if one is not specified by the configuration already
        """
        if flag in self.config.netperf_options:
            if not replace or value is None:
                return
            index = self.config.netperf_options.index(flag)
            self.config.netperf_options[index + 1] = value
        else:
            opt_item = [flag]
            if value is not None:
                opt_item.append(value)
            self.config.netperf_options = opt_item + self.config.netperf_options

    def configure(self, opts):
        opts = super().configure(opts)
        if self.has_pmc and self.has_pmc_profclock:
            self.logger.error("PMC and PMC_PROFCLOCK_STACKSAMPLE can not be used at the same time")
            raise ValueError("Configuration error")

        # Resolve binaries here as the configuration is stable at this point
        rootfs_netperf_base = self.benchmark.cheribsd_rootfs_path / self.config.netperf_path
        rootfs_netperf_bin = rootfs_netperf_base / "netperf"
        rootfs_netserver_bin = rootfs_netperf_base / "netserver"
        # Paths relative to the remote root directory
        self.netperf_bin = Path("/") / rootfs_netperf_bin.relative_to(self.benchmark.cheribsd_rootfs_path)
        self.netserver_bin = Path("/") / rootfs_netserver_bin.relative_to(self.benchmark.cheribsd_rootfs_path)
        self.logger.debug("Using %s %s", self.netperf_bin, self.netserver_bin)
        if self.has_pmc and not self.has_qemu:
            self._set_netperf_option("-g", "all")
        elif self.has_qemu:
            self._set_netperf_option("-g", "qemu")
        elif self.has_pmc_profclock:
            self._set_netperf_option("-g", "profclock")
        return opts

    def configure_iteration(self, script, iteration):
        super().configure_iteration(script, iteration)
        if self.has_pmc:
            pmc = self.benchmark.get_dataset(DatasetName.PMC)
            pmc_output = pmc.iteration_output_file(iteration)
            pmc_remote_output = script.local_to_remote_path(pmc_output)
            self._set_netperf_option("-G", pmc_remote_output, replace=True)
        if self.has_pmc_profclock:
            pmc = self.benchmark.get_dataset(DatasetName.PMC_PROFCLOCK_STACKSAMPLE)
            pmc_output = pmc.iteration_output_guestfile(iteration)
            pmc_output_path = script.local_to_remote_path(pmc_output)
            self._set_netperf_option("-G", pmc_output_path, replace=True)

    def gen_pre_benchmark(self, script):
        super().gen_pre_benchmark(script)
        # Start netserver here so that it does not interfere with any pre-benchmark
        # mesurements
        # Note that running the main netserver under ktrace should not affect the benchmark
        # too much as it forks for each client connnection and ktrace does not follow it, so
        # the ktrace noise should be limited to the part of the benchmark that is not traced.
        # This will however introduce some noise in the kernel stats sampled before and after
        # the whole benchmark run.
        pidmap = self.benchmark.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        if pidmap and self.config.netserver_resolve_forks:
            netserver_cmd = "ktrace"
            netserver_options = ["-t", "c", self.netserver_bin] + self.config.netserver_options
        else:
            netserver_cmd = self.netserver_bin
            netserver_options = self.config.netserver_options
        self.netserver_task = script.gen_bg_cmd(netserver_cmd, netserver_options, env=self.run_env)
        script.gen_sleep(5)

    def gen_benchmark(self, script, iteration):
        super().gen_benchmark(script, iteration)
        outpath = self.iteration_output_file(iteration)
        extra_output = []
        pmc = self.benchmark.get_dataset(DatasetName.PMC)
        if pmc:
            extra_output.append(pmc.iteration_output_file(iteration))
        script.gen_cmd(self.netperf_bin,
                       self.config.netperf_options.copy(),
                       outfile=outpath,
                       extra_outfiles=extra_output,
                       env=self.run_env,
                       collect_pid=True)

    def gen_post_benchmark(self, script):
        super().gen_post_benchmark(script)
        script.gen_stop_bg_cmd(self.netserver_task)
        pidmap = self.benchmark.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        if pidmap and self.config.netserver_resolve_forks:
            # Grab the extra pids forked by netserver
            script.gen_cmd("kdump", ["-s"], outfile=self._kdump_output_path())

    def aggregate(self):
        super().aggregate()
        grouped = self.merged_df.groupby(self.dataset_id_columns())
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        super().post_aggregate()
        agg_df = self._add_delta_columns(self.agg_df)
        self.agg_df = self._compute_delta_by_dataset(agg_df)
