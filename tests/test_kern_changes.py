import json
import uuid
from unittest.mock import Mock, PropertyMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pycheribenchplot.compile_db import CompilationDBModel
from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.kernel_history.cheribsd_annotations import *
from pycheribenchplot.kernel_history.model import *


@pytest.fixture
def fake_kernel_changes(fake_simple_benchmark):
    """
    Produce a dataframe mimicking the output of CheriBSDKernelAnnotations.

    XXX this may possibly be randomised from the model
    """
    data = {
        "schema": {
            "fields": [{
                "name": "file",
                "type": "string"
            }, {
                "name": "updated",
                "type": "string"
            }, {
                "name": "target_type",
                "type": "string"
            }, {
                "name": "changes",
                "type": "string"
            }, {
                "name": "changes_purecap",
                "type": "string"
            }, {
                "name": "change_comment",
                "type": "string"
            }, {
                "name": "hybrid_specific",
                "type": "boolean"
            }],
            "primaryKey": ["file"]
        },
        "data": [
            {
                "file": "cheribsd/sys/fs/smbfs/smbfs_vfsops.c",
                "updated": "20221205",
                "target_type": "kernel",
                "changes": ["user_capabilities", "integer_provenance"],
                "changes_purecap": [],
                "change_comment": None,
                "hybrid_specific": False
            },
            {
                "file": "cheribsd/sys/netinet/udp_usrreq.c",
                "updated": "20221205",
                "target_type": "kernel",
                "changes": ["user_capabilities"],
                "changes_purecap": [],
                "change_comment": None,
                "hybrid_specific": True
            },
            {
                "file": "cheribsd/sys/kern/subr_pctrie.c",
                "updated": "20221205",
                "target_type": "kernel",
                "changes": [],
                "changes_purecap": ["kdb"],
                "change_comment": None,
                "hybrid_specific": False
            },
            {
                "file": "cheribsd/sys/aaa_file.c",
                "updated": "20231205",
                "target_type": "kernel",
                "changes": [],
                "changes_purecap": ["support"],
                "change_comment": None,
                "hybrid_specific": False
            },
        ]
    }

    df = pd.read_json(json.dumps(data), orient="table")
    return df.sort_index()


@pytest.fixture
def fake_cdb(fake_simple_benchmark):
    """
    Produce a fake compilation DB consistent with the fake_kernel_changes
    """
    data = {
        "uuid": [str(fake_simple_benchmark.uuid)] * 4,
        "g_uuid": [str(fake_simple_benchmark.g_uuid)] * 4,
        "iterations": [-1] * 4,
        "target": ["cheribsd"] * 4,
        "file": [
            "cheribsd/sys/fs/smbfs/smbfs_vfsops.c",
            "cheribsd/sys/netinet/udp_usrreq.c",
            "cheribsd/sys/kern/subr_pctrie.c",
            "cheribsd/sys/kern/subr_vmem.c",
        ]
    }
    return pd.read_json(json.dumps(data)).set_index(["uuid", "g_uuid", "iterations", "target"])


@pytest.fixture
def other_cdb(fake_cdb):
    df = fake_cdb.copy()
    df.iat[0, df.columns.get_loc("file")] = "cheribsd/sys/bbb_file.c"
    return df


def test_raw_dataset_validate(fake_kernel_changes, fake_simple_benchmark):
    """
    Check the kernel changes dataframe model
    """

    expect = fake_kernel_changes.reset_index()
    expect["change_comment"] = expect["change_comment"].astype(object)
    expect = expect.set_index(fake_kernel_changes.index.names)

    schema = CheriBSDAnnotationsModel.to_schema(fake_simple_benchmark.session)
    df = schema.validate(fake_kernel_changes)

    assert_frame_equal(df, expect)


def test_kernel_file_changes_union(mocker, fake_session, fake_kernel_changes, fake_cdb, other_cdb):
    """
    Verify that CheriBSDAnnotationsUnion produces the union of the annotations and
    compilation DB datasets.
    """
    # Prepare mocked dependencies
    mock_files_dep = mocker.patch.object(CheriBSDAnnotationsUnion, "load_annotations", new_callable=PropertyMock)
    mock_load_annotations_task = Mock()
    mock_load_annotations_task.df.get.return_value = fake_kernel_changes.copy()
    mock_files_dep.return_value = mock_load_annotations_task
    mock_cdb_dep = mocker.patch.object(CheriBSDAnnotationsUnion, "load_cdb", new_callable=PropertyMock)
    mock_cdb_task_0 = Mock()
    mock_cdb_task_1 = Mock()
    mock_cdb_task_0.df.get.return_value = fake_cdb.copy()
    mock_cdb_task_1.df.get.return_value = other_cdb.copy()
    mock_cdb_dep.return_value = [mock_cdb_task_0, mock_cdb_task_1]

    task = CheriBSDAnnotationsUnion(fake_session, AnalysisConfig())
    task.run()

    annotations = task.df.get()
    # Check that aaa_file.c was filtered out because it is not in the compilation DB dataset
    expect_files = {
        "cheribsd/sys/fs/smbfs/smbfs_vfsops.c", "cheribsd/sys/netinet/udp_usrreq.c", "cheribsd/sys/kern/subr_pctrie.c"
    }
    assert set(annotations.index.unique("file")) == expect_files

    cdb = task.compilation_db.get()
    expect_cdb_files = {
        "cheribsd/sys/fs/smbfs/smbfs_vfsops.c", "cheribsd/sys/netinet/udp_usrreq.c", "cheribsd/sys/kern/subr_pctrie.c",
        "cheribsd/sys/kern/subr_vmem.c", "cheribsd/sys/bbb_file.c"
    }
    assert set(cdb["file"]) == expect_cdb_files
    assert len(cdb) == len(expect_cdb_files)
