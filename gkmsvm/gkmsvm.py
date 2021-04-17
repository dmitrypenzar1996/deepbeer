from __future__ import annotations

import json
import random
import shlex
import shutil
import stat
import string
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import (Any, Callable, ClassVar, Generator, Generic, Optional, T,
                    Union)

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pandas.api.types import is_integer_dtype


def write_table(
    tb: pd.DataFrame, path: Union[str, Path], index: bool = False, **kwargs
):
    tb.to_csv(path, index=index, sep="\t", **kwargs)


def is_user_executable(path: Path) -> bool:
    return path.stat().st_mode & stat.S_IXUSR != 0


def run_cmd(cmd: str, *, timeout=None) -> subprocess.CompletedProcess:
    cmd = shlex.split(cmd)
    pr = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout
    )
    return pr


def random_string(
    length: int = 16, alphabet: str = string.ascii_letters + string.digits
) -> str:
    return "".join(random.choices(alphabet, k=length))


def id2kmer(kmer_id: int, k: int, alphabet: list[str] = ["A", "C", "G", "T"]) -> str:
    kmer = []
    size = len(alphabet)
    for i in range(k):
        kmer.append(alphabet[kmer_id % size])
        kmer_id //= size
    return "".join(reversed(kmer))


def reverse_complement(
    seq: str, alphabet: dict[str, str] = {"A": "T", "C": "G", "G": "C", "T": "A"}
) -> str:
    rev_compl = []
    for s in reversed(seq):
        rev_compl.append(alphabet[s])
    return "".join(rev_compl)


def kmers_generator(
    k: int, distinct_reversed: bool = False
) -> Generator[str, None, None]:
    if k <= 0:
        raise Exception(f"k must positive: {k}")
    kmers = set()
    for i in range(4 ** k):
        s = id2kmer(i, k)
        if s not in kmers:
            yield s
        kmers.add(s)
        if not distinct_reversed:
            rs = reverse_complement(s)
            kmers.add(rs)


def kmer_split(seq: str, k: int) -> Generator[str, None, None]:
    if k <= 0:
        raise Exception(f"k must be positive: {k}")
    if len(seq) < k:
        raise Exception(
            f"Length of sequence must be equal or greater than k, {k} vs {len(seq)}"
        )

    for i in range(0, len(seq) - k + 1, 1):
        yield seq[i:i + k]


class ConfigException(Exception):
    pass


@dataclass
class Config:
    def _infer_validations(self) -> list[Callable[[Config], None]]:
        validations = []
        for par in self.__dict__:
            val_name = f"validate_{par}"
            if hasattr(self, val_name):
                validations.append(val_name)

        for m in dir(self):
            if m.startswith("validate_") and m not in validations:
                validations.append(m)
        return validations

    def validate(self) -> None:
        for v in self._validations:
            getattr(self, v)()

    def save(self, config_path: Union[str, Path], exists_ok: bool = True) -> None:
        config_path = Path(config_path)

        dt = {
            key: value for key, value in self.__dict__.items() if key != "_validations"
        }
        if config_path.exists():
            if config_path.is_dir():
                raise ConfigException("Provided path is dir")
            if not exists_ok:
                raise ConfigException("Provided path exists")
        with open(config_path, "w") as output:
            json.dump(dt, output, indent=4)

    def set_nans(self) -> None:
        pass

    def __post_init__(self) -> None:
        self.set_nans()
        self._validations = self._infer_validations()
        self.validate()

    @classmethod
    def load(cls, config_path: [str, Path]) -> Config:
        cls.check_config_path(config_path)
        with open(config_path) as infile:
            dt = json.load(infile)
        return cls(**dt)

    @staticmethod
    def check_config_path(config_path: [str, Path]) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigException("Config path doesn't exists")
        if not config_path.is_file():
            raise ConfigException("Config path is not a file")


class GKM_SVM_Exception(Exception):
    pass


@dataclass
class GKM_Train_Config(Config):
    kernel: str = "wgkm"
    word_length: int = 11
    info_cols_cnt: int = 7
    max_mismatch: int = 3
    C: Optional[float] = None
    pos_weight: float = 1.0
    distinct_reversed: bool = False
    gamma: Optional[float] = None
    eps: float = 0.001
    initial_decay_value: Optional[int] = None  # 50
    half_decay_distance: Optional[float] = None  # 50
    n_procs: int = 1
    cache_size: float = 100
    use_shrinkage: bool = False
    verbosity: int = "info"
    gkmtrain_path: str = "/Users/dmitrypenzar/Desktop/develop/lsgkm/bin/gkmtrain"

    KERNELS: ClassVar[dict] = {
        "gapped-kmer": 0,
        "lmer_full_filter": 1,
        "lmer_truncated_filter": 2,
        "gkmrbf": 3,
        "wgkm": 4,
        "wgkmrbf": 5,
    }
    RBF_KERNELS: ClassVar[tuple] = ("gkmrbf", "wgkmrbf")
    WEIGHT_KERNELS: ClassVar[tuple] = ("wgkm", "wgkmrbf")
    VERBOSE_LEVELS: ClassVar[dict] = {
        "error": 0,
        "warning": 1,
        "info": 2,
        "debug": 3,
        "trace": 4,
    }

    DEFAULT_GAMMA: ClassVar[float] = 2.0
    DEFAULT_INITIAL_DECAY: ClassVar[int] = 50
    DEFAULT_HALF_DECAY: ClassVar[float] = 50.0
    DEFAULT_C: ClassVar[dict] = {
        "gapped-kmer": 1.0,
        "lmer_full_filter": 1.0,
        "lmer_truncated_filter": 1.0,
        "gkmrbf": 10.0,
        "wgkm": 1.0,
        "wgkmrbf": 10.0,
    }

    def validate_kernel(self) -> None:
        if self.kernel not in self.KERNELS:
            raise GKM_SVM_Exception(f"Wrong kernel name: {self.kernel}")

    def validate_word_length(self) -> None:
        if (
            not isinstance(self.word_length, int)
            or self.word_length < 3
            or self.word_length > 12
        ):
            raise GKM_SVM_Exception(
                f"Wrong word length: {self.word_length}. "
                f" Must be an integer in [3, 12]"
            )

    def validate_info_cols_cnt(self) -> None:
        if (
            not isinstance(self.info_cols_cnt, int)
            or self.info_cols_cnt < 1
            or self.info_cols_cnt > self.word_length
        ):
            raise GKM_SVM_Exception(
                f"Wrong number of informative columns {self.info_cols_cnt}. "
                f"Must be an integer in [1, {self.word_length}]"
            )

    def validate_max_mismatch(self) -> None:
        if (
            not isinstance(self.max_mismatch, int)
            or self.max_mismatch < 0
            or self.max_mismatch > 4
        ):
            raise GKM_SVM_Exception(
                f"Wrong number of max mismatches: {self.max_mismatch}. "
                f"Must be an integer in [0, 4]"
            )
        max_m = self.word_length - self.info_cols_cnt
        if self.max_mismatch > self.word_length - self.info_cols_cnt:
            raise GKM_SVM_Exception(
                f"Wrong number of max mismatches: {self.max_mismatch}. "
                f"Can't be greater then word_length - info_cols_cnt: {max_m}"
            )

    def validate_gamma(self) -> None:
        if self.kernel not in self.RBF_KERNELS:
            if self.gamma is not None:
                raise GKM_SVM_Exception(
                    f"Gamma can be set only for rbf kernels ({', '.join(self.RBF_KERNELS)}), "
                    f"not {self.kernel}"
                )
        else:
            if not isinstance(self.gamma, (float, int)) or self.gamma <= 0:
                raise GKM_SVM_Exception(
                    f"Wrong gamma value: {self.gamma}. Must be a float greater then 0"
                )

    def validate_initial_decay_value(self) -> None:
        if self.kernel not in self.WEIGHT_KERNELS:
            if self.initial_decay_value is not None:
                raise GKM_SVM_Exception(
                    f"Initial decay value can be set only for weight kernels: "
                    f"({', '.join(self.WEIGHT_KERNELS)}), not {self.kernel}"
                )
        else:
            if (
                not isinstance(self.initial_decay_value, int)
                or self.initial_decay_value <= 0
                or self.initial_decay_value >= 255
            ):
                raise GKM_SVM_Exception(
                    f"Wrong initial decay value: {self.initial_decay_value}. "
                    f"Must be an integer in [1, 255]"
                )

    def validate_half_decay_distance(self) -> None:
        if self.kernel not in self.WEIGHT_KERNELS:
            if self.half_decay_distance is not None:
                raise GKM_SVM_Exception(
                    f"Half decay value can be set only for weight kernels: "
                    f"({', '.join(self.WEIGHT_KERNELS)}), not {self.kernel}"
                )
        else:
            if (
                not isinstance(self.half_decay_distance, (float, int))
                or self.half_decay_distance <= 0
            ):
                raise GKM_SVM_Exception(
                    f"Wrong half decay distance: {self.half_decay_distance}. "
                    f"Must be a float greater then 0"
                )

    def validate_C(self) -> None:
        if not isinstance(self.C, (float, int)) or self.C < 0:
            raise GKM_SVM_Exception(
                f"Wrong C value: {self.C}. " f"Must be a float greater then 0"
            )

    def validate_precision(self) -> None:
        if not isinstance(self.eps, float) or self.eps <= 0:
            raise GKM_SVM_Exception(
                f"Wrong precision parameter epsilon value: {self.eps}. "
                f"Must be a float greater then 0"
            )

    def validate_positive_weight(self) -> None:
        if not isinstance(self.pos_weight, (float, int)) or self.pos_weight <= 0:
            raise GKM_SVM_Exception(
                f"Wrong weight value: {self.pos_weight}. "
                f"Must be a float greater then 0"
            )

    def validate_cache_size(self) -> None:
        if not isinstance(self.cache_size, (float, int)) or self.cache_size <= 0:
            raise GKM_SVM_Exception(
                f"Wrong cache size value: {self.cache_size}. "
                f"Must be a float greater then 0"
            )

    def validate_use_shrinkage(self) -> None:
        if not isinstance(self.use_shrinkage, bool):
            raise GKM_SVM_Exception(
                f"Wrong use_shrinkage value: {self.use_shrinkage}. " f"Must be boolean"
            )

    def validate_n_procs(self) -> None:
        if self.n_procs not in (1, 4, 16):
            raise GKM_SVM_Exception(
                f"Wrong n_procs value: {self.n_procs}. " f"Must be 1, 4 or 16"
            )

    def validate_verbosity(self) -> None:
        if self.verbosity not in self.VERBOSE_LEVELS:
            raise GKM_SVM_Exception(
                f"Wrong verbosity value: {self.verbosity}. "
                f"Must be one of {', '.join(self.VERBOSE_LEVELS)}"
            )

    def validate_distinct_reversed(self) -> None:
        if not isinstance(self.distinct_reversed, bool):
            raise GKM_SVM_Exception(
                f"Wrong distinct reversed value: {self.distinct_reversed}. "
                f"Must be boolean"
            )

    def validate_gkmtrain_path(self) -> None:
        path = Path(self.gkmtrain_path)
        if not path.exists():
            raise GKM_SVM_Exception(
                f"Wrong path to gkmtrain: {self.gkmtrain_path}. " f"Doesn't exist"
            )
        if not path.is_file() or not is_user_executable(path):
            raise GKM_SVM_Exception(
                f"Wrong path to gkmtrain: {self.gkmtrain_path}. "
                f"Must be executable by te current user"
            )

    def set_nans(self) -> None:
        if self.gamma is None and self.kernel in self.RBF_KERNELS:
            self.gamma = self.DEFAULT_GAMMA
        if self.initial_decay_value is None and self.kernel in self.WEIGHT_KERNELS:
            self.initial_decay_value = self.DEFAULT_INITIAL_DECAY
        if self.half_decay_distance is None and self.kernel in self.WEIGHT_KERNELS:
            self.half_decay_distance = self.DEFAULT_HALF_DECAY
        if self.C is None:
            self.validate_kernel()
            self.C = self.DEFAULT_C[self.kernel]

    def to_command_line(self) -> str:
        return f"""
{self.gkmtrain_path}
-t {self.KERNELS[self.kernel]}
-l {self.word_length}
-k {self.info_cols_cnt}
-d {self.max_mismatch}
{f'-g {self.gamma}' if self.gamma is not None else ''}
{f'-M {self.initial_decay_value}' if self.initial_decay_value is not None else ''}
{f'-H {self.half_decay_distance}' if self.half_decay_distance is not None else ''}
{'-R' if {self.distinct_reversed} else ''}
-c {self.C}
-e {self.eps}
-w {self.pos_weight}
-m {self.cache_size}
{"-s" if self.use_shrinkage else ""}
-v {self.VERBOSE_LEVELS[self.verbosity]}
-T {self.n_procs}
"""


@dataclass
class GKM_Predict_Config(Config):
    n_procs: int = 1
    verbosity: int = "info"
    gkmpredict_path: str = "/Users/dmitrypenzar/Desktop/develop/lsgkm/bin/gkmpredict"

    VERBOSE_LEVELS: ClassVar[dict] = {
        "error": 0,
        "warning": 1,
        "info": 2,
        "debug": 3,
        "trace": 4,
    }

    def validate_verbosity(self) -> None:
        if self.verbosity not in self.VERBOSE_LEVELS:
            raise GKM_SVM_Exception(
                f"Wrong verbosity value: {self.verbosity}. "
                f"Must be one of {', '.join(self.VERBOSE_LEVELS)}"
            )

    def validate_gkmtest_path(self) -> None:
        path = Path(self.gkmpredict_path)
        if not path.exists():
            raise GKM_SVM_Exception(
                f"Wrong path to gkmtrain: {self.gkmpredict_path}. " f"Doesn't exist"
            )
        if not path.is_file() or not is_user_executable(path):
            raise GKM_SVM_Exception(
                f"Wrong path to gkmtrain: {self.gkmpredict_path}. "
                f"Must be executable by te current user"
            )

    def validate_n_procs(self) -> None:
        if self.n_procs not in (1, 4, 16):
            raise GKM_SVM_Exception(
                f"Wrong n_procs value: {self.n_procs}. " f"Must be 1, 4 or 16"
            )

    def to_command_line(self) -> str:
        return f"""
{self.gkmpredict_path}
-v {self.VERBOSE_LEVELS[self.verbosity]}
-T {self.n_procs}
"""


@dataclass
class Filter(Generic[T]):
    _filters: list[Callable[[Filter, Any], bool]] = field(
        default=None, repr=False, init=False
    )

    def pass_filters(self, item: Any) -> bool:
        for filt in self.filters:
            if not getattr(self, filt)(item):
                return False
        return True

    def apply(self, it: Iterable[Any]) -> Generator[Any, None, None]:
        for item in it:
            if self.pass_filters(item):
                yield item

    def _infer_filters(self) -> list[Callable[[Filter, Any], bool]]:
        filters = []
        classes = self.__class__.__mro__
        for cls in reversed(classes):
            for m in dir(cls):
                if m.startswith("filter_") and m not in filters:
                    filters.append(m)

        return filters

    @property
    def filters(self):
        if self._filters is None:
            self._filters = self._infer_filters()
        return self._filters


@dataclass
class FASTAFilter(Filter[Union[SeqRecord, Seq, str]]):
    pass


@dataclass
class BEDFilter(Filter["pandas.core.frame.Pandas"]):
    pass


@dataclass
class GKM_FASTAFilter(FASTAFilter):
    max_length: int
    ALPHABET: ClassVar[Union[str, list[str], set[str]]] = "ATGCatgc"

    def filter_length(self, seq: Union[SeqRecord, Seq, str]) -> bool:
        if len(seq) > self.max_length:
            return False
        return True

    def filter_alphabet(self, seq: Union[SeqRecord, Seq, str]) -> bool:
        for n in seq:
            if n not in self.ALPHABET:
                return False
        return True


class GKM_BEDFilter(BEDFilter):
    HUMAN_AUTOSOMES: ClassVar[set[str]] = set([f"chr{i}" for i in range(1, 22)])

    def filter_chromosomes(self, row: "pandas.core.frame.Pandas") -> bool:
        if row.chr not in self.HUMAN_AUTOSOMES:
            return False
        return True


@dataclass
class Genome:
    _chromosomes: dict[str, SeqRecord] = field(repr=False)

    def __getitem__(self, key):
        return self._chromosomes[key]

    @classmethod
    def from_fasta_file(cls, fasta_path: Union[str, Path]):
        fasta_path = Path(fasta_path)
        chroms = SeqIO.to_dict(SeqIO.parse("genomes/hg38.fa", format="fasta"))
        return cls(chroms)


@dataclass
class BEDDataset:
    _table: pd.DataFrame = field(repr=False)
    _filter: BEDFilter

    def __len__(self):
        return self._table.shape[0]

    def __iter__(self):
        return self._filter.apply(self._table.itertuples())

    @staticmethod
    def _check_chr_column(table):
        if "chr" not in table:
            raise Exception(
                'File must contain "chr" column, explicitly named in the header'
            )
        for val in table["chr"]:
            if not isinstance(val, str):
                raise Exception('"chr" column must contain only strings')

    @staticmethod
    def _check_start_column(table):
        if "start" not in table:
            raise Exception(
                'File must contain "start" column, explicitly named in the header'
            )
        if not is_integer_dtype(table["start"].dtype):
            raise Exception('"start" column must contain only integers')

    @staticmethod
    def _check_end_column(table):
        if "end" not in table:
            raise Exception(
                'File must contain "end" column, explicitly named in the header'
            )
        if not is_integer_dtype(table["end"].dtype):
            raise Exception('"end" column must contain only integers')

    @classmethod
    def from_table(
        cls, table: Union[str, Path, pd.DataFrame], filter: Optional[BEDFilter] = None
    ):
        if not isinstance(table, pd.DataFrame):
            table = pd.read_table(table)
        cls._check_chr_column(table)
        cls._check_start_column(table)
        cls._check_end_column(table)
        if filter is None:
            filter = BEDFilter()

        return cls(table, filter)


@dataclass
class FASTA_Dataset:
    length: int
    path: Path

    def __len__(self):
        return self.length

    @staticmethod
    def get_sequential_namer(pref: str = "seq") -> Callable[[str], str]:
        ind = 0

        def sequential_namer(x: str) -> str:
            nonlocal ind
            ind += 1
            return f"{pref}_{ind}"

        return sequential_namer

    @staticmethod
    def get_id_namer() -> Callable[[str], str]:
        def id_namer(x: str) -> str:
            return x

        return id_namer

    @staticmethod
    def _gen_temp_path() -> Path:
        temp = tempfile.NamedTemporaryFile(mode="w+")
        return Path(temp.name)

    @classmethod
    def from_seqrecs(
        cls,
        seqrecs: Iterable[SeqRecord],
        path: Optional[Union[str, Path]] = None,
        fasta_filter: FASTAFilter = None,
    ):
        if path is None:
            path = cls._gen_temp_path()
        if fasta_filter is None:
            fasta_filter = FASTAFilter()

        filt_it = fasta_filter.apply(seqrecs)
        with open(path, "w") as handle:
            length = SeqIO.write(filt_it, handle, format="fasta")

        return cls(length, path)

    @classmethod
    def from_fasta_file(
        cls,
        fasta_path: Union[str, Path],
        dataset_path: Optional[Union[str, Path]] = None,
        fasta_filter: FASTAFilter = None,
    ) -> FASTA_Dataset:
        seq_it = SeqIO.parse(fasta_path)
        return cls.from_seqrecs(seq_it, dataset_path, fasta_filter)

    @classmethod
    def from_seqs(
        cls,
        seqs: Iterable[str],
        path: Optional[Union[str, Path]] = None,
        namer: Optional[Callable[[str], str]] = None,
        fasta_filter: FASTAFilter = None,
    ) -> FASTA_Dataset:
        if namer is None:
            namer = cls.get_sequential_namer()
        seq_it = (SeqRecord(seq, id=namer(seq)) for seq in seqs)

        return cls.from_seqrecs(seq_it, path, fasta_filter)

    @classmethod
    def from_seqs_names(
        cls,
        names_seq: Iterable[tuple[str, str]],
        path: Optional[Union[str, Path]] = None,
        fasta_filter: FASTAFilter = None,
    ) -> FASTA_Dataset:

        seq_it = (SeqRecord(seq, id=name) for name, seq in names_seq)
        return cls.from_seqrecs(seq_it, path, fasta_filter)

    @staticmethod
    def _extract_seqrec_from_bed(row, genome: Genome):
        seq = genome[row.chr][row.start:row.end].seq
        seqrec = SeqRecord(seq=seq, id=f"{row.chr}_{row.start}:{row.end}")
        return seqrec

    @classmethod
    def from_BEDDataset(
        cls,
        bed_dataset: Union[str, Path, BEDDataset],
        genome: Union[str, Path, Genome],
        out_path: Optional[Union[str, Path]] = None,
        filter: FASTAFilter = None,
    ) -> FASTA_Dataset:
        if not isinstance(genome, Genome):
            genome = Genome.from_fasta_file(genome)
        if not isinstance(bed_dataset, BEDDataset):
            bed_dataset = BEDDataset.from_table(bed_dataset)

        seq_it = (cls._extract_seqrec_from_bed(row, genome) for row in bed_dataset)
        return cls.from_seqrecs(seq_it, out_path, filter)

    @classmethod
    def from_gc_sampling(dataset: FASTA_Dataset) -> FASTA_Dataset:
        raise NotImplementedError()

    @classmethod
    def kmer_dataset(
        cls,
        kmer_size: int,
        distinct_reversed: bool = False,
        path: Optional[Union[str, Path]] = None,
        exists_ok=False,
    ) -> FASTA_Dataset:
        return cls.from_seqs(
            kmers_generator(kmer_size, distinct_reversed),
            path,
            namer=cls.get_id_namer(),
            filter=FASTAFilter(),
        )


@dataclass
class GKMSVM_Trainer:
    workdir: Path
    config: Optional[GKM_Train_Config] = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = GKM_Train_Config()

    def run(self, pos_path: Path, neg_path: Path, out_pref: str) -> Path:
        cmd = self._get_cmd(pos_path, neg_path, out_pref)
        pr = run_cmd(cmd)
        model_path = self._get_model_path(out_pref)
        self._check_run(pr, model_path)

        return model_path

    def _check_run(self, pr: subprocess.CompletedProcess, model_path: Path) -> None:
        log_path = self._get_log_path(model_path)
        stdout = self._store_runout(pr, log_path)
        if pr.returncode != 0:
            raise GKM_SVM_Exception(f"gkmtrain exited with error: {log_path}")

        std_low = stdout.lower()
        if "wrong" in std_low or "error" in std_low:
            raise GKM_SVM_Exception(f"gkmtrain exited with error: {log_path}")

        if "warn" in std_low:
            print(f"gkmtrain output contains warning: {log_path}", file=sys.stderr)

        if not model_path.exists():
            raise GKM_SVM_Exception(
                "For the uknown reason gkmtrain didn't create model file."
                f"Logfile: {log_path}"
            )

    def _get_cmd(self, pos_path: Path, neg_path: Path, out_pref: Path) -> str:
        cmd = self.config.to_command_line()
        cmd = f"{cmd}\n {pos_path} {neg_path} {out_pref}"
        return cmd

    def _get_model_path(self, out_pref: str) -> Path:
        return Path(f"{out_pref}.model.txt")

    def _store_runout(self, pr: subprocess.CompletedProcess, path: str) -> str:
        stdout = pr.stdout.decode()
        with open(path, "w") as f:
            f.write(stdout)
        return stdout

    def _get_log_path(self, output_file: Path):
        log_path = self.workdir / f"{output_file.name}.train.log"
        ind = 1
        while log_path.exists():
            ind += 1
            log_path = self.workdir / f"{output_file.name}_{ind}.train.log"
        return log_path


@dataclass
class GKMSVM_Predictor:
    workdir: Path
    config: Optional[GKM_Predict_Config] = None

    def __post_init__(self):
        if self.config is None:
            self.config = GKM_Predict_Config()

    def run(self, model_path: Path, test_seq_file: Path, output_file: Path) -> Path:
        cmd = self._get_cmd(model_path, test_seq_file, output_file)
        pr = run_cmd(cmd)
        self._check_run(pr, output_file)

        return output_file

    def _get_cmd(self, model_path: Path, test_seq_file: Path, output_file: Path) -> str:
        cmd = self.config.to_command_line()
        cmd = f"{cmd}\n {test_seq_file} {model_path} {output_file}"
        return cmd

    def _check_run(self, pr: subprocess.CompletedProcess, pred_path: str) -> None:
        log_path = self._get_log_path(pred_path)
        stdout = self._store_runout(pr, log_path)

        if pr.returncode != 0:
            raise GKM_SVM_Exception(
                f"gkmpredict exited with error. Program stdout is at {log_path}"
            )

        std_low = stdout.lower()
        if "wrong" in std_low or "error" in std_low:
            raise GKM_SVM_Exception(
                f"gkmpredict exited with error. Program stdout is at {log_path}"
            )

        if "warn" in std_low:
            print(f"gkmtrain output contains warning: {log_path}", file=sys.stderr)

        if not pred_path.exists():
            raise GKM_SVM_Exception(
                f"For the uknown reason gkmpredict didn't create predictions file. Log: {log_path}"
            )

    def _get_log_path(self, output_file: Path):
        log_path = self.workdir / f"{output_file.name}.predict.log"
        ind = 1
        while log_path.exists():
            ind += 1
            log_path = self.workdir / f"{output_file.name}_{ind}.predict.log"
        return log_path

    def _store_runout(self, pr: subprocess.CompletedProcess, path: Path) -> str:
        stdout = pr.stdout.decode()
        with open(path, "w") as f:
            f.write(stdout)
        return stdout


@dataclass
class GKMSVM:
    model_path: Union[None, Path] = None
    model_tag: Union[None, str] = None

    trainer: GKMSVM_Trainer = field(init=False)
    predictor: GKMSVM_Predictor = field(init=False)
    workdir: Union[None, Path] = field(init=False)
    _tmp_dir: Union[None, str] = field(default=None, init=False)

    train_config: InitVar[Optional[Union[GKM_Train_Config, Path, str]]] = None
    predict_config: InitVar[Optional[Union[GKM_Predict_Config, Path, str]]] = None
    rootdir: InitVar[Optional[Union[str, Path]]] = None
    exist_ok: InitVar[bool] = True
    rm_if_exist: InitVar[bool] = False

    def fit(
        self, positive_dataset: FASTA_Dataset, negative_dataset: FASTA_Dataset
    ) -> GKMSVM:
        if self.fitted:
            self._restore()
        out_pref = self._get_model_path_pref(self.workdir)
        self.model_path = self.trainer.run(
            positive_dataset.path, negative_dataset.path, out_pref
        )
        return self

    def predict(
        self, dataset: FASTA_Dataset, return_path: bool = False
    ) -> Union[pd.DataFrame, Path]:
        if not self.fitted:
            raise GKM_SVM_Exception("Model is not trained")
        pred_path = self._get_prediction_path()
        self.predictor.run(self.model_path, dataset.path, pred_path)
        if return_path:
            return pred_path
        return pd.read_table(pred_path, header=None, names=["name", "score"])

    def score_kmers(self, k: int):
        if k < self.trainer.config.word_length:
            raise GKM_SVM_Exception(
                f"Can't score kmers for k ({k}) < "
                f"word_length ({self.trainer.config.word_length})"
            )
        ds = FASTA_Dataset.kmer_dataset(
            k, distinct_reversed=self.trainer.config.distinct_reversed
        )
        return self.predict(ds, return_path=True)

    def _restore(self) -> None:
        if self.fitted:
            self.model_path.unlink()

    @classmethod
    def load(cls, workdir: Union[str, Path]) -> GKMSVM:
        workdir = Path(workdir)
        tr_cfg = cls._get_train_config_path(workdir)
        pr_cfg = cls._get_predict_config_path(workdir)
        model_path = cls._get_model_path(workdir)
        if not model_path.exists():
            model_path = None
        model_tag = workdir.name
        rootdir = workdir.parent

        return cls(
            train_config=tr_cfg,
            predict_config=pr_cfg,
            model_path=model_path,
            model_tag=model_tag,
            rootdir=rootdir,
            exist_ok=True,
            rm_if_exist=False,
        )

    @staticmethod
    def _gen_model_tag(root_dir: Path):
        model_tag = random_string()
        while (root_dir / model_tag).exists():
            model_tag = random_string()
        return model_tag

    def _check_workdir(self, exist_ok: bool, rm_if_exist: bool) -> None:
        if self.workdir.exists():
            if not exist_ok:
                raise GKM_SVM_Exception("Provided dir already exists")
            if not self.workdir.is_dir():
                raise GKM_SVM_Exception("Provided path is not a dir")
            if rm_if_exist:
                shutil.rmtree(self.workdir)

    def _init_workdir(self, rootdir: Optional[Path], exist_ok: bool, rm_if_exist: bool):
        if rootdir is not None:
            if self.model_tag is None:
                model_tag = self._gen_model_tag(rootdir)
                self.model_tag = model_tag
            self.workdir = rootdir / self.model_tag
            self._check_workdir(exist_ok=exist_ok, rm_if_exist=rm_if_exist)
            self.workdir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_dir = tempfile.TemporaryDirectory()
            if self.model_tag is None:
                self.model_tag = "model"
            self.workdir = Path(self._tmp_dir.name) / self.model_tag

    def __post_init__(
        self,
        train_config: Optional[Union[GKM_Train_Config, Path, str]],
        predict_config: Optional[Union[GKM_Predict_Config, Path, str]],
        rootdir: Optional[Union[str, Path]],
        exist_ok: bool,
        rm_if_exist: bool,
    ) -> None:

        if rootdir is not None:
            rootdir = Path(rootdir)
        self._init_workdir(rootdir, exist_ok=exist_ok, rm_if_exist=rm_if_exist)

        if train_config is None:
            train_config = GKM_Train_Config()
        elif isinstance(train_config, (Path, str)):
            train_config = GKM_Train_Config.load(train_config)

        if predict_config is None:
            predict_config = GKM_Predict_Config()
        if isinstance(predict_config, (Path, str)):
            predict_config = GKM_Predict_Config.load(predict_config)

        self.trainer = GKMSVM_Trainer(config=train_config, workdir=self.workdir)
        self.predictor = GKMSVM_Predictor(
            config=predict_config, workdir=self._get_predictions_dir()
        )
        self._save_cfg()

    def _save_cfg(self) -> None:
        tr_path = self._get_train_config_path(self.workdir)
        self.trainer.config.save(tr_path)
        pr_path = self._get_predict_config_path(self.workdir)
        self.predictor.config.save(pr_path)

    @property
    def fitted(self) -> bool:
        return self.model_path is not None

    @classmethod
    def _get_model_path_pref(cls, workdir: Path) -> Path:
        return workdir / "gkm"

    @classmethod
    def _get_model_path(cls, workdir) -> Path:
        return Path(f"{cls._get_model_path_pref(workdir)}.model.txt")

    @staticmethod
    def _get_train_config_path(workdir: Path) -> Path:
        return workdir / "train.cfg"

    @staticmethod
    def _get_predict_config_path(workdir: Path) -> Path:
        return workdir / "predict.cfg"

    def _get_predictions_dir(self) -> Path:
        path = self.workdir / "predictions"
        if not path.exists():
            path.mkdir(parents=True)
        return path

    def _get_prediction_path(self) -> Path:
        pred_dir = self._get_predictions_dir()
        name = f"{random_string()}.prediction.tab"
        while (pred_dir / name).exists():
            name = f"{random_string()}.prediction.tab"
        return pred_dir / name


@dataclass
class SVMDelta:
    scores: dict[str, float] = field(repr=False)
    k: int
    distinct_reversed: bool = False

    def __post_init__(self) -> None:
        self._check_scores()

    def _check_scores(self) -> None:
        kmers_cnt = 4 ** self.k
        if kmers_cnt != len(self.scores):
            raise GKM_SVM_Exception(
                "Scores doesn't include information about all kmers."
                f"Entries: {len(self.scores)}. Required: {kmers_cnt}"
            )

    def _check_seq(self, seq) -> None:
        for s in seq:
            if s not in ("A", "T", "G", "C"):
                raise GKM_SVM_Exception(
                    f"Sequence must contain only A, T, G and C: {s}"
                )

    def decision_function(self, seq: str) -> float:
        self._check_seq(seq)
        score = 0

        for kmer in kmer_split(seq, self.k):
            score += self.scores[kmer]
        return score

    def _trunc_seq(
        self, seq: Union[str, SeqRecord], pos: int
    ) -> tuple[Union[str, SeqRecord], int]:
        ss = max(0, pos - self.k + 1)
        se = min(len(seq), pos + self.k)
        return seq[ss:se], pos - ss

    @staticmethod
    def _get_alt_seq(seq: str, pos: int, alt: str):
        return seq[:pos] + alt + seq[pos + 1:]

    def score_snv(self, seq: str, pos: int, alt: str) -> float:
        seq, pos = self._trunc_seq(seq, pos)
        ref_seq = seq
        alt_seq = self._get_alt_seq(seq, pos, alt)
        ref_score = self.decision_function(ref_seq)
        alt_score = self.decision_function(alt_seq)
        return alt_score - ref_score

    def _check_save_path(path: Path, exists_ok: bool = True) -> None:
        if path.exists():
            if not exists_ok:
                raise GKM_SVM_Exception(f"Provided path exists: {path}")
            if not path.is_file():
                raise GKM_SVM_Exception(f"Provided path is not a file: {path}")

    def save(self, path: Union[Path, str], exist_ok=True) -> None:
        path = Path(path)
        self._check_save_path(path)
        with open(path, "w") as out:
            for kmer, score in self.scores.items():
                print(f"{kmer}\t{score}", file=out)

    @staticmethod
    def _check_load_path(path: Path) -> None:
        if not path.exists():
            raise GKM_SVM_Exception(f"Provided path doesn't exist: {path}")
        if not path.is_file():
            raise GKM_SVM_Exception(f"Provided path is not a file: {path}")

    @classmethod
    def load(cls, path: Union[Path, str], distinct_reversed=False) -> SVMDelta:
        path = Path(path)
        cls._check_load_path(path)
        dt = {}
        with open(path, "r") as k_scores:
            seq, _ = k_scores.readline().split()
            k = len(seq)
            k_scores.seek(0)

            for ind, line in enumerate(k_scores):
                seq, score = line.split()
                score = float(score)
                if len(seq) != k:
                    raise GKM_SVM_Exception(
                        f"Provided file contains kmers of different size: line {ind}. "
                        f"Expected {k}, got {len(seq)}"
                    )
                dt[seq] = score
                if not distinct_reversed:
                    rseq = reverse_complement(seq)
                    dt[rseq] = score
        return cls(dt, k, distinct_reversed=distinct_reversed)
