# Introduction
The CHERI benchplot tool/library is an attempt to build a flexible system to run CHERI experiments on CheriBSD and process the results.
Ideally the structure is modular and reusable enough that it can be adapted to speed-up data analysis or used as a starting point
for custom data analysis and visualization.

## Installation

The library has a number of python dependencies plus some C++ native modules for dwarf parsing.
The latter requires the Cheri LLVM libraries to be installed in the cherisdk directory and the `boost::python` library.
Cheri-benchplot requires a recent python version, it has been tested with Python 3.10+.

The following commands should install the dependencies and the pycheribenchplot package:
```
# This must point to the root of the cherisdk, containing the rootfs and the 'sdk' directory.
$ export CHERISDK=/path/to/cherisdk
# It is recommended to setup a virtual environment
$ virtualenv my-env
$ source my-env/bin/activate
(my-env) $ pip install .
# To check that the installation succeeded:
(my-env) $ benchplot-cli.py -h
```

It is possible to run tests as follows:
```
(my-env) $ pytest tests
```

## Design
The tool uses two configuration files:
 - User configuration: by default in `~/.config/cheri-benchplot.json`, specifies cheri SDK information.
 - Pipeline configuration: specifies the data-generation tasks and cheribsd platform combinations to run.

Currently the runner only supports starting up CHERI-QEMU instances for behavioural analysis and qemu tracing.
There is some support for the VCU118 FPGA via cheribuild scripts.

The main entry point is the `benchplot-cli` tool. This provides a number of subcommands:
```
(my-env) $ benchplot-cli.py
usage: benchplot-cli.py [-h] [-v] [-l LOGFILE] [-w WORKERS] [-c CONFIG] {session,info} ...

CHERI plot and data analysis tool benchplot-cli

positional arguments:
  {session,info}        command
    session             Manage and ispect sessions.
    info                Display information about tasks.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose output
  -l LOGFILE, --logfile LOGFILE
                        logfile
  -w WORKERS, --workers WORKERS
                        Override max number of workers from configuration file
  -c CONFIG, --config CONFIG
                        User environment configuration file. Defaults to ~/.config/cheri-benchplot.json
```

The cheri-benchplot library operates around the notion of a *session*.
A session is described by a _pipeline configuration file_ (see the json files in `pipelines/...`), which provides information about
the tasks that generate the data and the analysis steps to generate output artefacts.
When the session is created, the pipeline configuration is parsed and the combinations of platform configurations, parameterization and
data generation tasks are resolved to produce a _benchmark matrix_ that represents all the data generation tasks that will produce inputs
for the analysis.
The benchmark matrix for a session can be displayed with `benchplot-cli.py info session path/to/session`.

The session supports two main activities:
    - Run: will execute all data generation tasks described by the benchmark matrix, along with any dependencies.
    - Analyse: will run all the analysis tasks configured to produce the desired outputs.

Tasks are logically organised into namespaces. This is used to filter tasks that are compatible with each other.
For example, analysis tasks in the `foo` namespace should all be able to operate on data produced by generators in the `foo` namespace.

At session creation, a new `Benchmark` object is created for each benchmark/instance combination.
This represents a data-generation run and is uniquely identified by an UUID.
The benchmark instance will manage the data-generation and analysis for a benchmark/instance configuration pair.
Each benchmark contains a group of tasks (`ExecutionTask` subclasses) that represent data sources.
These tasks are responsible for generating the commands to produce the data, extract the data from the target cheribsd (or other) instance
once the benchmark is done, and load the data for analysis.

## Example

 - XXX Rework this example to make it simpler.

Required branches:
 - cheribsd: statcounters-update
 - qemu: qemu-experimental-tracing
 - netperf: cheri-netperf
 - cheri-perfetto: cheri-perfetto
 - cheri-benchplot: master
 
The example runs a simple UDP_RR netperf benchmark and collects data from all supported data sources.
 1. Build cheribsd: `cheribuild cheribsd-riscv64-purecap --cheribsd/build-alternate-abi-kernels --cheribsd/build-bench-kernels`
 2. Build qemu: `cheribuild qemu --qemu/configure-options '--enable-perfetto-log-instr'`
 3. Build netperf: `cheribuild netperf-riscv64-purecap`
 4. Build cheri-perfetto: this is not currently automated by cheribuild.
 ```
 cd path/to/cheri-perfetto
 mkdir -p out
 tools/install-build-deps
 # Can either use the shipped gn/ninja or the system gn/ninja
 tools/gn gen out/cheri-perfetto
 tools/ninja -C out/cheri-perfetto
 ```
 7. Install benchplot and the perfetto python bindings (might be best to use a virtualenv)
 ```
 pip install -e path/to/cheri-benchplot
 pip install -e path/to/cheri-perfetto/src/trace_processor/python
 ```
 5. Make sure the benchplot user configuration in `.config/cheri-benchplot.json` matches the your setup:
```
{
    "sdk_path": "path/to/cherisdk",
    "perfetto_path": "path/to/cheri-perfetto/out/cheri-perfetto",
    "cheribuild_path": "path/to/cheribuild/cheribuild.py"
}
```
 6. Run the demo session:
 ```
 python benchplot.py demo/demo-netperf-multi-instance/benchplot_config.json run
 python benchplot.py demo/demo-netperf-multi-instance/benchplot_config.json analyse
 ```

## Other tracing backends
CHERI-QEMU supports two other tracing backends that may be useful
 - protobuf backend:
   + Build qemu with `--enable-protobuf-log-instr`, run with `--cheri-trace-backend protobuf`
   + Emits a stream of protobufs that represent the trace entries. See `cheri-perfetto/protos/perfetto/trace/track_event/qemu_log_entry.proto` for the proto definitions.
 - json backend:
   + Build qemu with `--enable-json-log-instr`, run with `--cheri-trace-backend json`
   + Emit a json file that contains the log entries. The format should be more stable and easier to parse than the text format, however protobufs or perfetto should
   be used for any trace that is not a toy program.

## CHERI Trace Processor
This is currently a skeleton for more advanced parsing of qemu traces from both the protobuf and the perfetto backends.
It contains an example to read a protobuf trace from the protobuf backend. Perfetto traces should be loaded using the
`trace_processor_shell` from the cheri-perfetto repository.

### Example setup to record and observe protobuf traces
Required branches:
 - cheribsd: statcounters-update (should also work on dev, with less data output)
 - qemu: qemu-experimental-tracing
 - cheri-benchplot: master
 
 1. Build qemu `cheribuild qemu --qemu/configure-options '--enable-protobuf-log-instr'`
 2. Run qemu (assuming you have built cheribsd and a disk image) `cheribuild run-riscv64-purecap --run-riscv64-purecap/extra-options '--cheri-trace-backend protobuf'`
 3. In the qemu guest, run a sample program:

     ```sh
     $ sysctl hw.qemu_trace_perthread=1
     $ qtrace exec helloworld
     $ poweroff
     ```
     This will generate the qemu-protobuf.pb file (currently hardcoded, sorry I am lazy...)
 4. Run `python cheri_trace_processor/cheri_trace_processor.py path/to/qemu-protobuf.pb`
