import dataclasses
import glob
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import prefect
from prefect import Flow, Parameter, Task, task, unmapped
from prefect.engine import signals

DNASE_MASC2_CALLPEAK_CFG = dict(
    macs2_path="/home/penzard/miniconda3/envs/macs2/bin/macs2",
    genome_size="hs",
    pval_thresh=0.01,
    nomodel=True,
    shift=75,
    extsize=150,
    bdg=False,
    peak_format="narrowPeak",
    keep_dup="all",
)

DNASE_IDR_CFG = dict(
    idr_ths=1, rank="p.value", plot=True, idr_path="/home/penzard/miniconda3/bin/idr"
)

SAMTOOLS_FLAGSTAT_CHECK_CFG = dict(
    samtools_path="/home/penzard/bio/bin/bin/samtools", threads=5
)


def run_cmd(cmd, *, timeout=None):
    cmd = shlex.split(cmd)
    pr = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    return pr


def report2logger(pr_result, logger):
    if pr_result.returncode != 0:
        msg_fn = logger.error
    else:
        msg_fn = logger.debug

    stdout = pr_result.stdout.decode()
    if stdout:
        msg_fn(f"STDOUT:\n{stdout}")
    stderr = pr_result.stderr.decode()
    if stderr:
        msg_fn(f"STDERR:\n{stderr}")


@dataclass
class BamFlagStatistic:
    passed: int
    failed: int

    @property
    def total(self):
        return self.passed + self.failed


@dataclass
class BamFlagStatOutput:
    total: BamFlagStatistic  # total number of reads
    secondary: BamFlagStatistic  # number of secondary alignments
    supplementary: BamFlagStatistic  # number of supplementary alignments
    duplicates: BamFlagStatistic  # number of duplicates
    mapped: BamFlagStatistic  # number of mapped reads
    paired: BamFlagStatistic  # number of reads with pair
    read1: BamFlagStatistic  # number of first reads in pair
    read2: BamFlagStatistic  # number of second reads in pair
    properly_paired: BamFlagStatistic  # number of properly paired reads (orientation and inset size are reasonable)
    both_mapped: BamFlagStatistic  # both itself and mate mapped
    singletons: BamFlagStatistic  # read mapped, but mate is not
    different_chr: BamFlagStatistic  # mapped with mate to the different chromosomes
    different_chr_qge5: BamFlagStatistic  # mapped with mate to the different chromosomes and MAPQ >= 5
    FIELDS_MAPPING: ClassVar[dict] = {
        "total": "total",
        "secondary": "secondary",
        "supplementary": "supplementary",
        "duplicates": "duplicates",
        "mapped": "mapped",
        "paired in sequencing": "paired",
        "read1": "read1",
        "read2": "read2",
        "properly paired": "properly_paired",
        "with itself and mate mapped": "both_mapped",
        "singletons": "singletons",
        "with mate mapped to a different chr": "different_chr",
        "with mate mapped to a different chr (mapQ >= 5)": "different_chr_qge5",
    }
    PASSED_FIELD: ClassVar[str] = "QC-passed reads"
    FAILED_FIELD: ClassVar[str] = "QC-failed reads"

    @classmethod
    def parse(cls, output):
        dt = json.loads(output)
        obj_dt = {}
        for field, name in cls.FIELDS_MAPPING.items():
            passed = dt[cls.PASSED_FIELD][field]
            failed = dt[cls.FAILED_FIELD][field]
            obj_dt[name] = BamFlagStatistic(passed, failed)
        return cls(**obj_dt)

    def generate_warning(self):
        warnings = []
        if self.paired.total - self.properly_paired.total > 0:
            pss = self.paired.passed - self.properly_paired.passed
            fail = self.paired.failed - self.properly_paired.failed
            msg = (
                f"BAM contains reads not properly paired, {pss} "
                f"passing quality threshold, {fail} others"
            )
            warnings.append(msg)
        # noinspection DuplicatedCode
        if self.secondary.total > 0:
            msg = (
                f"BAM contains secondary alignments, {self.secondary.passed} "
                f"passing quality threshold, {self.secondary.failed} others"
            )
            warnings.append(msg)
        if self.supplementary.total > 0:
            msg = (
                f"BAM contains supplementary alignments, {self.supplementary.passed} "
                f"passing quality threshold, {self.supplementary.failed} others"
            )
            warnings.append(msg)

        # noinspection DuplicatedCode
        if self.duplicates.total > 0:
            msg = (
                f"BAM contains duplicates, {self.duplicates.passed} "
                f"passing quality threshold, {self.duplicates.failed} others"
            )
            warnings.append(msg)
        if self.singletons.total > 0:
            msg = (
                f"BAM contains singletons, {self.singletons.passed} "
                f"passing quality threshold, {self.singletons.failed} others"
            )
            warnings.append(msg)
        if self.different_chr.total > 0:
            msg = (
                f"BAM contains reads with mate paired to a different chromosomee, {self.different_chr.passed} "
                f"passing quality threshold, {self.different_chr.failed} others"
            )
            warnings.append(msg)

        return "\n".join(warnings)

    def warning(self, logger):
        w = self.generate_warning()
        if w:
            logger.warning(w)
            return 1
        return 0


def make_samtools_flagstat_cmd(*, bam_path, threads, samtools_path):
    cmd = f"""
    {samtools_path} flagstat {bam_path}
    --threads {threads}
    --output-fmt json
    """
    return cmd


class BAMChecker(Task):
    def __init__(self, samtools_cfg, **kwargs):
        self.samtools_cfg = samtools_cfg
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def run(self, bam_path):
        cmd = make_samtools_flagstat_cmd(bam_path=bam_path, **self.samtools_cfg)
        result = run_cmd(cmd)
        report2logger(result, self.logger)
        if result.returncode != 0:
            raise signals.FAIL()
        parsed_r = BamFlagStatOutput.parse(result.stdout.decode())
        parsed_r.warning(self.logger)

        return parsed_r


# https://docs.google.com/document/d/1e3cCormg0SnQW6zr7VYBWvXC1GiPc5GSy80qlKBPwlA/edit#


def make_macs2_callpeak_cmd(
    *,
    treatment_files,
    out_peak_pref,
    outdir,
    fmt,
    control_files,
    genome_size,
    macs2_path,
    pval_thresh,
    nomodel,
    shift,
    extsize,
    bdg,
    peak_format,
    keep_dup,
):
    if isinstance(treatment_files, list):
        treatment_files = " ".join(treatment_files)
    if isinstance(control_files, list):
        control_files = " ".join(control_files)

    if peak_format == "narrowPeak":
        broad = False
    elif peak_format == "broadPeak":
        broad = True
    else:
        raise Exception("Wrong peak format")

    cmd = f"""{macs2_path} callpeak 
    --treatment {treatment_files} 
    {f"--control {control_files}" if control_files is not None else ""}
    --outdir {outdir}
    --format {fmt}
    --shift {shift}  
    --extsize  {extsize}
    --pvalue {pval_thresh}
    --gsize {genome_size} 
    --name {out_peak_pref}
    {"--bdg" if bdg else ""}
    {"--nomodel" if nomodel else ""}
    {"--broad" if broad else ""}
    --keep-dup {keep_dup}
    """

    return cmd


@dataclass
class Masc2CallPeakOutput:
    peak_path: str
    masc_out: str
    _q75_peak_length: int = dataclasses.field(default=None, init=False, repr=False)
    _peaks_table: pd.DataFrame = dataclasses.field(default=None, init=False, repr=False)

    NARROW_PEAK_COLUMNS: ClassVar[list] = [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue",  # overall enrichment for the region
        "-log10(pvalue)",
        "-log10(qvalue)",  # FDR controlled
        "center",  # peak center
    ]

    BROAD_PEAK_COLUMNS: ClassVar[list] = [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue",  # overall enrichment for the region
        "-log10(pvalue)",
        "-log10(qvalue)",  # FDR controlled
    ]

    MIN_PEAKS: ClassVar[int] = 10 ** 5
    MAX_PEAKS: ClassVar[int] = 10 ** 6

    MAX_PEAKS_Q75_LENGTH: ClassVar[int] = 1500

    @property
    def peaks_count(self):
        return self.peaks_table.shape[0]

    @property
    def q75_peak_length(self):
        if self._q75_peak_length is None:
            self._q75_peak_length = np.quantile(self.peaks_table["length"], 0.75)
        return self._q75_peak_length

    def generate_warnings(self):
        warnings = []
        if self.peaks_count < self.MIN_PEAKS:
            msg = f"The number of called peaks is less than {self.MIN_PEAKS}: {self.peaks_count}"
            warnings.append(msg)
        elif self.peaks_count > self.MAX_PEAKS:
            msg = f"The number of called peaks is greater than {self.MAX_PEAKS}: {self.peaks_count}"
            warnings.append(msg)

        if self.q75_peak_length > self.MAX_PEAKS_Q75_LENGTH:
            msg = f"The 75% quantile peak length is greater than {self.MAX_PEAKS_Q75_LENGTH}: {self.q75_peak_length}"
            warnings.append(msg)

        return "\n".join(warnings)

    def warning(self, logger):
        w = self.generate_warnings()
        if w:
            logger.warning(w)
            return 1
        return 0

    @property
    def fmt(self):
        return self.peak_path.rsplit(".", 1)[-1]

    @property
    def colnames(self):
        if self.fmt == "narrowPeak":
            return self.NARROW_PEAK_COLUMNS
        elif self.fmt == "broadPeak":
            return self.BROAD_PEAK_COLUMNS
        raise Exception("Invalid file format")

    @property
    def peaks_table(self):
        if self._peaks_table is None:
            tb = pd.read_table(self.peak_path)
            tb.columns = self.colnames  # to avoid table truncating
            tb["length"] = tb["end"] - tb["start"]
            self._peaks_table = tb

        return self._peaks_table


def macs2callpeak_peak_path(outdir, out_peak_pref, peak_format):
    peaks_path = os.path.join(outdir, f"{out_peak_pref}_peaks.{peak_format}")
    return peaks_path


class Macs2CallPeak(Task):
    def __init__(self, macs2callpeak_cfg, **kwargs):
        self.macs2callpeak_cfg = macs2callpeak_cfg
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def run(self, *, treatment_files, control_files, fmt, outdir, out_peak_pref):
        cmd = make_macs2_callpeak_cmd(
            treatment_files=treatment_files,
            control_files=control_files,
            fmt=fmt,
            outdir=outdir,
            out_peak_pref=out_peak_pref,
            **self.macs2callpeak_cfg,
        )
        result = run_cmd(cmd)
        report2logger(result, self.logger)
        if result.returncode != 0:
            raise signals.FAIL()

        peaks_path = macs2callpeak_peak_path(
            outdir, out_peak_pref, self.macs2callpeak_cfg["peak_format"]
        )

        parsed_r = Masc2CallPeakOutput(peaks_path, result.stdout.decode())
        parsed_r.warning(self.logger)

        return parsed_r


@dataclass
class IDROutput:
    peaks_path: str
    peak_fmt: str
    replic_cnt: int
    idr_out: str
    _q75_peak_length: int = dataclasses.field(default=None, init=False, repr=False)
    _peaks_table: pd.DataFrame = dataclasses.field(default=None, init=False, repr=False)

    REGULAR_PEAK_COLUMNS: ClassVar[dict] = {
        "narrowPeak": [
            "chr",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalValue",  # overall enrichment for the region
            "-log10(pvalue)",
            "-log10(qvalue)",  # FDR controlled
            "center",  # peak center
            "localIDR",
            "globalIDR",
        ],
        "broadPeak": [
            "chr",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalValue",  # overall enrichment for the region
            "-log10(pvalue)",
            "-log10(qvalue)",  # FDR controlled
            "localIDR",
            "globalIDR",
        ],
    }

    PER_REPLIC_COLUMNS: ClassVar[dict] = {
        "narrowPeak": ["start", "end", "signalValue", "center"],
        "broadPeak": ["start", "end", "signalValue"],
    }

    POSITIVE_IDR_THR: ClassVar[int] = 0.05
    NEGATIVE_IDR_THR: ClassVar[int] = 0.25

    MIN_POSITIVE_SIZE: ClassVar[int] = 10 ** 4
    MAX_POSITIVE_SIZE: ClassVar[int] = 10 ** 5

    MIN_NEGATIVE_SIZE: ClassVar[int] = 10 ** 4
    MAX_NEGATIVE_SIZE: ClassVar[int] = 10 ** 5

    MAX_PEAKS_Q75_LENGTH: ClassVar[int] = 1500

    @property
    def colnames(self):
        regular = self.REGULAR_PEAK_COLUMNS[self.peak_fmt]
        per_replic = self.PER_REPLIC_COLUMNS[self.peak_fmt]
        cols = regular.copy()
        for i in range(1, self.replic_cnt + 1):
            rep_cols = [f"{name}_{i}" for name in per_replic]
            cols.extend(rep_cols)

        return cols

    @property
    def peaks_table(self):
        if self._peaks_table is None:
            tb = pd.read_table(self.peaks_path)
            tb.columns = self.colnames  # to avoid table truncating
            tb["length"] = tb["end"] - tb["start"]
            self._peaks_table = tb
        return self._peaks_table

    @property
    def q75_peak_length(self):
        if self._q75_peak_length is None:
            self._q75_peak_length = np.quantile(self.peaks_table["length"], 0.75)
        return self._q75_peak_length

    def generate_warnings(self):
        warnings = []
        pos = self.positive_dataset()

        # noinspection DuplicatedCode
        if pos.shape[0] < self.MIN_POSITIVE_SIZE:
            msg = f"The number of positive examples is less than {self.MIN_POSITIVE_SIZE}: {pos.shape[0]}"
            warnings.append(msg)
        elif pos.shape[0] > self.MAX_POSITIVE_SIZE:
            msg = f"The number of positive examples is greater than {self.MAX_POSITIVE_SIZE}: {pos.shape[0]}"
            warnings.append(msg)

        neg = self.negative_dataset()

        # noinspection DuplicatedCode
        if neg.shape[0] < self.MIN_NEGATIVE_SIZE:
            msg = f"The number of negative examples is less than {self.MIN_NEGATIVE_SIZE}: {neg.shape[0]}"
            warnings.append(msg)
        elif neg.shape[0] > self.MAX_POSITIVE_SIZE:
            msg = f"The number of negative examples is greater than {self.MAX_NEGATIVE_SIZE}: {neg.shape[0]}"
            warnings.append(msg)

        if self.q75_peak_length > self.MAX_PEAKS_Q75_LENGTH:
            msg = f"The 75% quantile peak length is greater than {self.MAX_PEAKS_Q75_LENGTH}: {self.q75_peak_length}"
            warnings.append(msg)

        return "\n".join(warnings)

    def warning(self, logger):
        w = self.generate_warnings()
        if w:
            logger.warning(w)
            return 1
        return 0

    def positive_dataset(self, idr_threshold=None):
        idr_threshold = idr_threshold or self.POSITIVE_IDR_THR
        score_ths = IDROutput.idr_to_score(idr_threshold)
        return self.peaks_table[self.peaks_table["score"] >= score_ths]

    def negative_dataset(self, idr_threshold=None):
        idr_threshold = idr_threshold or self.NEGATIVE_IDR_THR
        score_ths = IDROutput.idr_to_score(idr_threshold)
        return self.peaks_table[self.peaks_table["score"] <= score_ths]

    @staticmethod
    def idr_to_score(p):
        return min(int(-125 * np.log2(p)), 1000)


def make_idr_command(
    *, peak_files, oracle_path, out_file, peak_fmt, idr_ths, rank, plot, idr_path
):
    cmd = f"""
    {idr_path} --samples {' '.join(peak_files)}
    --peak-list {oracle_path}
    --input-file-type {peak_fmt}
    --output-file {out_file}
    --idr-threshold {idr_ths}
    --rank {rank}
    {"--plot" if plot else ""}
    """
    return cmd


class IDR(Task):
    def __init__(self, idr_cfg, **kwargs):
        self.idr_cfg = idr_cfg
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def run(self, *, peaks, oracle, out_dir, peak_fmt):
        peak_files = [p.peak_path for p in peaks]
        oracle_path = oracle.peak_path if oracle else None

        out_file = os.path.join(out_dir, f"idr.{peak_fmt}")
        cmd = make_idr_command(
            peak_files=peak_files,
            out_file=out_file,
            oracle_path=oracle_path,
            peak_fmt=peak_fmt,
            **self.idr_cfg,
        )

        result = run_cmd(cmd)
        report2logger(result, self.logger)
        if result.returncode != 0:
            raise signals.FAIL()

        parsed_r = IDROutput(out_file, peak_fmt, 2, result.stderr.decode())
        parsed_r.warning(self.logger)

        return parsed_r


@task(name="Infer bam format")
def infer_bam_format(check1, check2):
    if check1.paired.total == 0 and check2.paired.total == 0:
        fmt = "BAM"
    elif check1.paired.total != 0 and check2.paired.total != 0:
        fmt = "BAMPE"
    else:  # one replic is paired and one is unpaired
        logger = prefect.context.get("logger")
        logger.warning("Replics have different formats")
        fmt = "BAM"
    return fmt


@task(name="Locate experiment files")
def find_exp_files(exp_name, exp_type, root_dir):
    src_mask = os.path.join(root_dir, exp_type, f"{exp_name}_[0-9]*.bam")
    exp_files = glob.glob(src_mask)
    if not exp_files:
        raise signals.FAIL("No expeeriment files found")
    return exp_files


@task(name="Link files to the destination dir")
def link_files(files, dest_dir, base_name, name_template_fn, rm_if_exist):
    dirpath = os.path.join(dest_dir, base_name)
    dirpath = os.path.abspath(dirpath)
    if os.path.exists(dirpath) and rm_if_exist:
        shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok=True)

    paths = []
    for r_ind, bam_path in enumerate(files, 1):
        r_name = name_template_fn(bam_path)
        r_path = os.path.join(dirpath, r_name)
        os.symlink(bam_path, r_path)
        paths.append(r_path)

    return dirpath, paths


# noinspection PyArgumentList
@task
def get_list_index(x, ind):
    return x[ind]


# no *args supported
# noinspection PyArgumentList
@task
def combine2_in_list(x, y):
    return [x, y]


def create_dnase_process_flow():
    with Flow("experiment") as flow:
        root_dir = Parameter("root_dir")
        run_dir = Parameter("run_dir")
        exp_type_lst = Parameter("exp_type")
        align_name_lst = Parameter("align_name")

        exp_files_lst = find_exp_files.map(
            align_name_lst, exp_type_lst, unmapped(root_dir)
        )

        all_paths = link_files.map(
            exp_files_lst,
            unmapped(run_dir),
            align_name_lst,
            unmapped(lambda x: os.path.basename(x)),
            rm_if_exist=unmapped(True),
        )

        outdirs = get_list_index.map(all_paths, unmapped(0))
        bam_links = get_list_index.map(all_paths, unmapped(1))
        replics1 = get_list_index.map(bam_links, unmapped(0))
        replics2 = get_list_index.map(bam_links, unmapped(1))

        checker1 = BAMChecker(SAMTOOLS_FLAGSTAT_CHECK_CFG, name="check replic1")
        check1 = checker1.map(replics1)

        checker2 = BAMChecker(SAMTOOLS_FLAGSTAT_CHECK_CFG, name="check replic2")
        check2 = checker2.map(replics2)

        fmt = infer_bam_format.map(check1, check2)

        callpeak1 = Macs2CallPeak(DNASE_MASC2_CALLPEAK_CFG, name="callpeak replic1")
        peak1 = callpeak1.map(
            treatment_files=replics1,
            control_files=unmapped(None),
            fmt=fmt,
            outdir=outdirs,
            out_peak_pref=unmapped("1"),
        )

        callpeak2 = Macs2CallPeak(DNASE_MASC2_CALLPEAK_CFG, name="callpeak replic2")
        peak2 = callpeak2.map(
            treatment_files=replics2,
            control_files=unmapped(None),
            fmt=fmt,
            outdir=outdirs,
            out_peak_pref=unmapped("2"),
        )

        callpeak_joined = Macs2CallPeak(
            DNASE_MASC2_CALLPEAK_CFG, name="callpeak joined"
        )
        oracle = callpeak_joined.map(
            treatment_files=bam_links,
            control_files=unmapped(None),
            fmt=fmt,
            outdir=outdirs,
            out_peak_pref=unmapped("joined"),
        )

        peaks = combine2_in_list.map(peak1, peak2)
        idr = IDR(DNASE_IDR_CFG, name="idr joining")

        idr.map(
            peaks=peaks,
            oracle=oracle,
            out_dir=outdirs,
            peak_fmt=unmapped(DNASE_MASC2_CALLPEAK_CFG["peak_format"]),
        )

    return flow
