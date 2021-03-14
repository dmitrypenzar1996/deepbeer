import os
import glob
import shutil
import shlex
import subprocess
import json
import dataclasses
import numpy as np 
import pandas as pd

import prefect
from prefect import Task, task
from prefect import Flow, Parameter
from prefect.engine import signals

from dataclasses import dataclass
from typing import ClassVar

DNASE_MASC2_CALLPEAK_CFG = dict(
    macs2_path="/home/penzard/miniconda3/envs/macs2/bin/macs2",
    genome_size="hs",
    pval_thresh=0.01,
    nomodel=True,
    shift=75,
    extsize=150,
    bdg=False,
    peak_format="narrowPeak",
    keep_dup="all")

DNASE_IDR_CFG = dict(idr_ths=1,
                    rank="p.value",
                    plot=True,
                    idr_path="/home/penzard/miniconda3/bin/idr")

SAMTOOLS_FLAGSTAT_CHECK_CFG = dict(
    samtools_path="/home/penzard/bio/bin/bin/samtools",
    threads=10)

def run_cmd(cmd, *, timeout=None):
    cmd = shlex.split(cmd)
    pr = subprocess.run(cmd,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        timeout=timeout)
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
class BamFlagStatOutput:
    @dataclass
    class BamFlagStatistic:
        passed: int
        failed: int
            
        @property
        def total(self):
            return self.passed + self.failed
            
    total: 'BamFlagStatLine' # total number of reads
    secondary: 'BamFlagStatLine' # number of secondary alignments 
    supplementary: 'BamFlagStatLine' # number of supplementary alignments 
    duplicates: 'BamFlagStatLine' # number of duplicates
    mapped: 'BamFlagStatLine' # number of mapped reads 
    paired: 'BamFlagStatLine' # number of reads with pair 
    read1: 'BamFlagStatLine' # number of first reads in pair
    read2: 'BamFlagStatLine' # number of second reads in pair
    properly_paired: 'BamFlagStatLine' # number of properly paired reads (orientation and inset size are reasonable)
    both_mapped: 'BamFlagStatLine' # both itself and mate mapped 
    singletons: 'BamFlagStatLine' # read mapped, but mate is not
    different_chr: 'BamFlagStatLine' # mapped with mate to the different chromosomes
    different_chr_qge5: 'BamFlagStatLine' # mapped with mate to the different chromosomes and MAPQ >= 5
    FIELDS_MAPPING: ClassVar[dict] =  {'total': 'total',
                           'secondary': 'secondary',
                           'supplementary': 'supplementary',
                           'duplicates': 'duplicates',
                           'mapped': 'mapped',
                           'paired in sequencing': 'paired',
                           'read1': 'read1',
                           'read2': 'read2',
                           'properly paired': 'properly_paired',
                           'with itself and mate mapped': 'both_mapped',
                           'singletons': 'singletons',
                           'with mate mapped to a different chr': 'different_chr',
                           'with mate mapped to a different chr (mapQ >= 5)': 'different_chr_qge5'
                          }
    PASSED_FIELD: ClassVar[str] = 'QC-passed reads'
    FAILED_FIELD: ClassVar[str] = 'QC-failed reads'

    @classmethod
    def parse(cls, output):
        dt = json.loads(output)
        obj_dt = {}
        for field, name in cls.FIELDS_MAPPING.items():
            passed = dt[cls.PASSED_FIELD][field]
            failed = dt[cls.FAILED_FIELD][field]
            obj_dt[name] = cls.BamFlagStatistic(passed, failed)
        return cls(**obj_dt)
    
    
    def generate_warning(self):
        warnings = []
        if self.paired.total - self.properly_paired.total > 0:
            pss = self.paired.passed - self.properly_paired.passed
            fail = self.paired.failed - self.properly_paired.failed
            msg = f"BAM contains reads not properly paired, {pss} "\
                  f"passing quality threshold, {fail} others"
            warnings.append(msg)
        if self.secondary.total > 0:
            msg = f"BAM contains secondary alignments, {self.secondary.passed} "\
                  f"passing quality threshold, {self.secondary.failed} others"
            warnings.append(msg)
        if self.supplementary.total > 0:
            msg = f"BAM contains supplementary alignments, {self.supplementary.passed} "\
                  f"passing quality threshold, {self.supplementary.failed} others"
            warnings.append(msg)
        if self.duplicates.total > 0:
            msg = f"BAM contains duplicates, {self.duplicates.passed} "\
                  f"passing quality threshold, {self.duplicates.failed} others"
            warnings.append(msg)
        if self.singletons.total > 0:
            msg = f"BAM contains singletons, {self.singletons.passed} "\
                  f"passing quality threshold, {self.singletons.failed} others"
            warnings.append(msg)
        if self.different_chr.total > 0:
            msg = f"BAM contains reads with mate paired to a different chromosomee, {self.different_chr.passed} "\
                  f"passing quality threshold, {self.different_chr.failed} others"
            warnings.append(msg)
            
        return "\n".join(warnings)
    
    
    def warning(self, logger):
        w = self.generate_warning()
        if w:
            logger.warning(w)
            return 1
        return 0
        
        

def make_samtools_flagstat_cmd(*,
                               bam_path, 
                               threads,
                               samtools_path):
    cmd = f'''
    {samtools_path} flagstat {bam_path}
    --threads {threads}
    --output-fmt json
    '''
    return cmd

class BAMChecker(Task):
    def __init__(self, samtools_cfg, **kwargs):
        self.samtools_cfg = samtools_cfg
        super().__init__(**kwargs)
    
    def run(self, bam_path):
        cmd = make_samtools_flagstat_cmd(bam_path=bam_path, **self.samtools_cfg)
        result = run_cmd(cmd)
        report2logger(result, self.logger)
        if result.returncode != 0:
            raise signals.FAIL()    
        parsed_r = BamFlagStatOutput.parse(result.stdout.decode())
        parsed_r.warning(self.logger)
        
        return parsed_r

#https://docs.google.com/document/d/1e3cCormg0SnQW6zr7VYBWvXC1GiPc5GSy80qlKBPwlA/edit#

def make_macs2_callpeak_cmd(*,
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
                        keep_dup):
    
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
    
    cmd = f'''{macs2_path} callpeak 
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
    '''  
    
    return cmd

@dataclass
class Masc2CallPeakOutput:
    peak_path: str
    masc_out: str
    _median_peak_length: int = dataclasses.field(default=None, init=False, repr=False)
    _peaks_table: pd.DataFrame = dataclasses.field(default=None, init=False, repr=False)

    NARROW_PEAK_COLUMNS: ClassVar[list] = [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue", # overall enrichment for the region
        "-log10(pvalue)", 
        "-log10(qvalue)", # FDR controlled
        "center", # peak center
    ]
        
    BROAD_PEAK_COLUMNS: ClassVar[list] =  [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue", # overall enrichment for the region
        "-log10(pvalue)", 
        "-log10(qvalue)", # FDR controlled
    ]
        
    MIN_PEAKS: ClassVar[int] = 10 ** 5
    MAX_PEAKS: ClassVar[int] = 10 ** 6 
        
    MAX_PEAKS_MEDIAN_LENGTH: ClassVar[int] = 1500
    
    @property
    def peaks_count(self):
        return self.peaks_table.shape[0]
    
    @property
    def median_peak_length(self):
        if self._median_peak_length is None:
            self._median_peak_length = np.median(self.peaks_table['length'])
        return self._median_peak_length
    
    def generate_warnings(self):
        warnings = []
        if self.peaks_count < self.MIN_PEAKS:    
            msg = f"The number of called peaks is less than {self.MIN_PEAKS}: {self.peaks_count}"
            warnings.append(msg)
        elif self.peaks_count > self.MAX_PEAKS:
            msg = f"The number of called peaks is more than {self.MAX_PEAKS}: {self.peaks_count}"
            warnings.append(msg)
        
        if self.median_peak_length > self.MAX_PEAKS_MEDIAN_LENGTH:
            msg = f"The median peak length is greater than {self.MAX_PEAKS_MEDIAN_LENGTH}: {self.median_peak_length}"
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
        return self.peak_path.rsplit('.', 1)[-1]
    
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
            tb.columns = self.colnames # to avoid table truncating 
            tb['length'] = tb['end'] - tb['start']
            self._peaks_table = tb
            
        return self._peaks_table
    
def macs2callpeak_peak_path(outdir, out_peak_pref, peak_format):
    peaks_path = os.path.join(outdir, 
                              f"{out_peak_pref}_peaks.{peak_format}")
    return peaks_path


class Macs2CallPeak(Task):
    def __init__(self, 
                 macs2callpeak_cfg,
                 **kwargs):
        self.macs2callpeak_cfg = macs2callpeak_cfg
        super().__init__(**kwargs)
        
        
    def run(self, *, treatment_files, 
            control_files,
            fmt,
            outdir,
            out_peak_pref):
        cmd = make_macs2_callpeak_cmd(treatment_files=treatment_files,
                                      control_files=control_files,
                                      fmt=fmt, 
                                      outdir=outdir,
                                      out_peak_pref=out_peak_pref,
                                      **self.macs2callpeak_cfg)
        result = run_cmd(cmd)
        report2logger(result, self.logger)
        if result.returncode != 0:
            raise signals.FAIL()    
        
        peaks_path = macs2callpeak_peak_path(outdir,
                                             out_peak_pref,
                                             self.macs2callpeak_cfg['peak_format'])
            
        parsed_r = Masc2CallPeakOutput(peaks_path, result.stdout.decode())
        parsed_r.warning(self.logger)
        
        return parsed_r
    
@dataclass
class IDROutput:
    peaks_path: str
    peak_fmt: str
    replic_cnt: int
    idr_out: str 
    _peaks_count: int = dataclasses.field(default=None, init=False, repr=False)
    _peaks_table: pd.DataFrame = dataclasses.field(default=None, init=False, repr=False)
        
    REGULAR_PEAK_COLUMNS: ClassVar[dict] = {
        'narrow': [
            "chr",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalValue", # overall enrichment for the region
            "-log10(pvalue)", 
            "-log10(qvalue)", # FDR controlled
            "center", # peak center
            "localIDR",
            "globalIDR"],
        "broad": [
            "chr",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalValue", # overall enrichment for the region
            "-log10(pvalue)", 
            "-log10(qvalue)", # FDR controlled
            "localIDR",
            "globalIDR"]
    }
        
    PER_REPLIC_COLUMNS: ClassVar[dict] = {
        "narrow":["start",
                  "end",
                  "signalValue",
                  "center"],
        "broad":["start",
                  "end",
                  "signalValue"]
    }
        
    POSITIVE_IDR_THR: ClassVar[int] = 0.20
    NEGATIVE_IDR_THR: ClassVar[int] = 0.50

    MIN_POSITIVE_SIZE: ClassVar[int] = 10 ** 4
    MAX_POSITIVE_SIZE: ClassVar[int] = 10 ** 5
        
    MIN_NEGATIVE_SIZE: ClassVar[int] = 10 ** 4
    MAX_NEGATIVE_SIZE: ClassVar[int] = 10 ** 5
    
    MAX_PEAKS_MEDIAN_LENGTH: ClassVar[int] = 1500

    @property
    def colnames(self):
        regular = self.REGULAR_PEAK_COLUMNS[self.peak_fmt]
        per_replic = self.PER_REPLIC_COLUMNS[self.peak_fmt]
        cols = regular.copy()
        for i in range(1, self.replic_cnt+1):
            rep_cols = [f"{name}_{i}" for name in per_replic]
            cols.extend(rep_cols)
            
        return cols
    
    @property
    def peaks_table(self):
        if self._peaks_table is None:
            tb = pd.read_table(self.peaks_path)
            tb.columns = self.colnames # to avoid table truncating 
            self._peaks_table = tb
        return self._peaks_table
    
    @property
    def median_peak_length(self):
        if self._median_peak_length is None:
            self._median_peak_length = np.median(self.peaks_table['length'])
        return self._median_peak_length
    
    def generate_warnings(self):
        warnings = []
        pos = self.positive_dataset()
        if pos.shape[0] < self.MIN_POSITIVE_SIZE:
            msg = f"The number of positive examples is less than {self.DEFAULT_MIN_POSITIVE_SIZE}"
            warnings.append(msg)
        elif pos.shape[0] > self.MAX_POSITIVE_SIZE:
            msg = f"The number of negative examples is less than {self.DEFAULT_MAX_POSITIVE_SIZE}"
            warnings.append(msg)
            
        neg = self.negative_dataset()
        if neg.shape[0] < self.MIN_NEGATIVE_SIZE:
            msg = f"The number of negative examples is less than {self.DEFAULT_MIN_NEGATIVE_SIZE}"
            warnings.append(msg)
        elif neg.shape[0] > self.MAX_POSITIVE_SIZE:
            msg = f"The number of negative examples is less than {self.DEFAULT_MIN_NEGATIVE_SIZE}"
            warnings.append(msg)
            
        if self.median_peak_length(self) > self.MAX_PEAKS_MEDIAN_LENGTH:
            msg = f"The median peak length is greater than {self.MAX_PEAKS_MEDIAN_LENGTH}: {self.median_peak_length}"
            warnings.append(msg)
        
        return "\n".join(warnings)
        
    
    def warning(self, logger):
        w = self.generate_warnings()
        if w:
            logger.warning(w)
            return 1
        return 0
    
    def positive_dataset(self, idr_threshold=None):
        idr_threshold = idr_threshold or DEFAULT_POSITIVE_IDR_THR
        ths = IDROutput.prop_to_idrcolval(idr_threshold)
        return self.peaks_table[self.peaks_table.globalIDR >= ths]
    
    def negative_dataset(self, idr_threshold=None):
        idr_threshold = idr_threshold or DEFAULT_POSITIVE_IDR_THR
        ths = IDROutput.prop_to_idrcolval(idr_threshold)
        return self.peaks_table[self.peaks_table.globalIDR <= ths]
    
    
    @staticmethod
    def prop_to_idrcolval(p):
        return min(int(-125 * np.log2(p)),
                   1000)
    
def make_idr_command(*,
                     peak_files,
                     oracle_path,
                     out_file,
                     peak_fmt,
                     idr_ths,
                     rank,
                     plot,
                     idr_path):
    cmd = f'''
    {idr_path} --samples {' '.join(peak_files)}
    --peak-list {oracle_path}
    --input-file-type {peak_fmt}
    --output-file {out_file}
    --idr-threshold {idr_ths}
    --rank {rank}
    {"--plot" if plot else ""}
    '''
    return cmd 


class IDR(Task):
    def __init__(self, idr_cfg, **kwargs):
        self.idr_cfg = idr_cfg
        super().__init__(**kwargs)
        
    def run(self, 
            *, 
            peaks,
            oracle,
            out_dir,
            peak_fmt):
        
        peak_files = [p.peak_path for p in peaks]
        oracle_path = oracle.peak_path if oracle else None
        
        out_file = os.path.join(out_dir, f"idr.{peak_fmt}")
        cmd = make_idr_command(peak_files=peak_files,
                              out_file=out_file,
                              oracle_path=oracle_path,
                              peak_fmt=peak_fmt,
                              **self.idr_cfg)
    
        result = run_cmd(cmd)
        report2logger(result, self.logger)
        if result.returncode != 0:
            raise signals.FAIL()    
                    
        parsed_r = IDROutput(out_file, peak_fmt, 2, result.stdout.decode())
        parsed_r.warning(self.logger)
        
        return parsed_r
    
@task(name="Infer bam format")
def infer_bam_format(check1, check2):
    if check1.paired.total == 0 and check2.paired.total == 0:
        fmt = "BAM"
    elif check1.paired.total != 0 and check2.paired.total != 0:
        fmt = "BAMPE"
    else:
        raise signals.FAIL
    return fmt

@task(name="Locate experiment files")
def find_exp_files(exp_name, exp_type, root_dir):
    src_mask = os.path.join(root_dir, exp_type, f"{exp_name}_[0-9]*.bam")
    exp_files = glob.glob(src_mask)
    if not exp_files:
        raise signals.FAIL
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
        
    return paths

def create_dnase_process_flow():
    with Flow("experiment") as flow:
        root_dir = Parameter('root_dir')
        run_dir = Parameter('run_dir')
        exp_type = Parameter('exp_type')
        align_name = Parameter('align_name')

        exp_files = find_exp_files(align_name, 
                                   exp_type, 
                                   root_dir)

        bam_links = link_files(exp_files,
                           run_dir,  
                           align_name, 
                           lambda x: os.path.basename(x), 
                            rm_if_exist=True)


        checker1 = BAMChecker(SAMTOOLS_FLAGSTAT_CHECK_CFG, name="check replic1")  
        check1 = checker1(bam_links[0])

        checker2 = BAMChecker(SAMTOOLS_FLAGSTAT_CHECK_CFG, name="check replic2")  
        check2 = checker2(bam_links[1])

        fmt = infer_bam_format(check1, check2)

        callpeak1 = Macs2CallPeak(DNASE_MASC2_CALLPEAK_CFG, name="callpeak replic1")
        peak1 = callpeak1(treatment_files=bam_links[0], 
                 control_files=None,
                 fmt=fmt,
                 outdir=run_dir,
                 out_peak_pref="1")

        callpeak2 = Macs2CallPeak(DNASE_MASC2_CALLPEAK_CFG, name="callpeak replic2")
        peak2 = callpeak2(treatment_files=bam_links[1], 
                 control_files=None,
                 fmt=fmt,
                 outdir=run_dir,
                 out_peak_pref="2")

        callpeak_joined = Macs2CallPeak(DNASE_MASC2_CALLPEAK_CFG, name="callpeak joined")
        oracle = callpeak_joined(treatment_files=bam_links, 
                 control_files=None,
                 fmt=fmt,
                 outdir=run_dir,
                 out_peak_pref="joined")

        idr = IDR(DNASE_IDR_CFG, name="idr joining")


        idr(peaks=[peak1, peak2], 
            oracle=oracle,
            out_dir=run_dir, 
            peak_fmt=DNASE_MASC2_CALLPEAK_CFG["peak_format"])
        
    return flow
    