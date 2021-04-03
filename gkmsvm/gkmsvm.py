import subprocess
import shlex 
import os
import glob
import json

from dataclasses import dataclass, field
from typing import ClassVar
from pathlib import Path

class ConfigException(Exception):
    pass

@dataclass
class Config:  
    def _infer_validations(self):   
        validations = [] 
        for par in self.__dict__:
            val_name = f"validate_{par}"
            if hasattr(self, val_name):
                validations.append(val_name)

        for m in dir(self):
            if m.startswith("validate_") and m not in validations:
                validations.append(m)
        return validations
    
    def validate(self):
        for v in self._validations:
            getattr(self, v)()
                        
    def save_config(self, config_path):
        dt = {key:value for key, value in self.__dict__.items() if key != "_validations"}
        with open(config_path, 'w') as output:
            json.dump(dt, output, indent=4)
            
    def set_nans(self):
        pass
    
    def __post_init__(self):  
        self.set_nans()
        self._validations = self._infer_validations()
        self.validate()
        
    @classmethod
    def load_config(cls, config_path):
        cls.check_config_path(config_path)
        with open(config_path) as infile:
            dt = json.load(infile)
        return cls(**dt)
    
    @staticmethod
    def check_config_path(config_path):
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigException("Config path doesn't exists")
        if not config_path.is_file():
            raise ConfigException("Config path is not a file")

class GKM_SVM_Exception(Exception):
    pass 

@dataclass
class GKM_Config(Config):
    kernel: str = "wgkm"
    word_length: int = 11
    info_cols_cnt: int = 7
    max_mismatch: int = 3
    C: float = None
    pos_weight: float = 1.0
    distinct_reversed: bool = False 
    gamma: float = None
    eps: float = 0.001
    initial_decay_value: int = None # 50
    half_decay_distance: float = None # 50
    n_procs: int = 1
    cache_size: float = 100
    use_shrinkage: bool = False
    verbosity: int = "info"
    
    KERNELS: ClassVar[dict] = {
        "gapped-kmer": 0,
        "lmer_full_filter": 1,
        "lmer_truncated_filter": 2,
        "gkmrbf": 3,
        "wgkm": 4,
        "wgkmrbf": 5
    }
    RBF_KERNELS: ClassVar[tuple] = ("gkmrbf",
                  "wgkmrbf")
    WEIGHT_KERNELS: ClassVar[tuple] = ("wgkm", 
                      "wgkmrbf")
    VERBOSE_LEVELS: ClassVar[dict] = {
        "error": 0,
        "warning": 1,
        "info": 2,
        "debug": 3,
        "trace": 4}
        
    DEFAULT_GAMMA: ClassVar[float] = 2.0
    DEFAULT_INITIAL_DECAY: ClassVar[int] = 50
    DEFAULT_HALF_DECAY: ClassVar[float] = 50.0
    DEFAULT_C: ClassVar[dict] = {
        "gapped-kmer": 1.0,
        "lmer_full_filter": 1.0,
        "lmer_truncated_filter": 1.0,
        "gkmrbf": 2.0,
        "wgkm": 1.0,
        "wgkmrbf": 2.0
    } 
    
    def validate_kernel(self):
        if self.kernel not in self.KERNELS:
            raise GKM_SVM_Exception(f"Wrong kernel name: {self.kernel}")

    def validate_word_length(self):
        if not isinstance(self.word_length, int) or self.word_length < 3 or self.word_length > 12:
            raise GKM_SVM_Exception(f"Wrong word length: {self.word_length}."
                                    f" Must be an integer in [3, 12]")

    def validate_info_cols_cnt(self):
        if not isinstance(self.info_cols_cnt, int) or self.info_cols_cnt < 1 or self.info_cols_cnt > self.word_length:
            raise GKM_SVM_Exception(f"Wrong number of informative columns {self.info_cols_cnt}."
                                    f"Must be an integer in [1, {self.word_length}]")

    def validate_max_mismatch(self):
        if not isinstance(self.max_mismatch, int) or self.max_mismatch < 0 or self.max_mismatch > 4:
            raise GKM_SVM_Exception(f"Wrong max mismatches: {self.max_mismatch}."
                                    f"Must be an integer in [0, 4]")

    def validate_gamma(self):
        if not self.kernel in self.RBF_KERNELS:
            if self.gamma is not None:
                raise GKM_SVM_Exception(f"Gamma can be set only for rbf kernels ({', '.join(self.RBF_KERNELS)}), "
                                        f"not {self.kernel}")
        else:
            if not isinstance(self.gamma, (float, int)) or self.gamma <= 0:
                raise GKM_SVM_Exception(f"Wrong gamma value: {self.gamma}. Must be a float greater then 0")

    def validate_initial_decay_value(self):
        if not self.kernel in self.WEIGHT_KERNELS:
            if self.initial_decay_value is not None:
                raise GKM_SVM_Exception(f"Initial decay value can be set only for weight kernels: "
                                        f"({', '.join(self.WEIGHT_KERNELS)}), not {self.kernel}")
        else:
            if not isinstance(self.initial_decay_value, int) or self.initial_decay_value <= 0\
              or self.initial_decay_value >= 255:
                raise  GKM_SVM_Exception(f"Wrong initial decay value: {self.initial_decay_value}. "
                                         f"Must be an integer in [1, 255]")

    def validate_half_decay_distance(self):
        if not self.kernel in self.WEIGHT_KERNELS:
            if self.half_decay_distance is not None:
                raise GKM_SVM_Exception(f"Half decay value can be set only for weight kernels: "
                                        f"({', '.join(self.WEIGHT_KERNELS)}), not {self.kernel}")
        else:
            if not isinstance(self.half_decay_distance, (float, int)) or self.half_decay_distance <= 0:
                raise GKM_SVM_Exception(f"Wrong half decay distance: {self.half_decay_distance}. "
                                        f"Must be a float greater then 0")

    def validate_C(self):
        if not isinstance(self.C, (float, int)) or self.C < 0:
            raise GKM_SVM_Exception(f"Wrong C value: {self.C}. "
                                    f"Must be a float greater then 0")

    def validate_precision(self):
        if not isinstance(self.eps, float) or self.eps <= 0:
            raise GKM_SVM_Exception(f"Wrong precision parameter epsilon value: {self.eps}. "
                                    f"Must be a float greater then 0")

    def validate_positive_weight(self):
        if not isinstance(self.pos_weight, (float, int)) or self.pos_weight <= 0:
            raise GKM_SVM_Exception(f"Wrong weight value: {self.pos_weight}. "
                                    f"Must be a float greater then 0")

    def validate_cache_size(self):
        if not isinstance(self.cache_size, (float, int)) or self.cache_size <= 0:
            raise GKM_SVM_Exception(f"Wrong cache size value: {self.cache_size}. "
                                    f"Must be a float greater then 0")

    def validate_use_shrinkage(self):
        if not isinstance(self.use_shrinkage, bool):
            raise GKM_SVM_Exception(f"Wrong use_shrinkage value: {self.use_shrinkage}. "
                                    f"Must be boolean")

    def validate_n_procs(self):
        if self.n_procs not in (1, 4, 16):
            raise GKM_SVM_Exception(f"Wrong n_procs value: {self.n_procs}. "
                                    f"Must be 1, 4 or 16")

    def validate_verbosity(self):
        if self.verbosity not in self.VERBOSE_LEVELS:
            raise GKM_SVM_Exception(f"Wrong verbosity value: {self.verbosity}"
                                    f"Must be one of {', '.join(self.VERBOSE_LEVELS)}")
            
    def validate_distinct_reversed(self):
        if not isinstance(self.distinct_reversed, bool):
            raise GKM_SVM_Exception(f"Wrong distinct reversed value: {self.distinct_reversed}. "
                                    f"Must be boolean")
            
    
    def set_nans(self):
        if self.gamma is None and self.kernel in self.RBF_KERNELS:
            self.gamma = self.DEFAULT_GAMMA
        if self.initial_decay_value is None and self.kernel in self.WEIGHT_KERNELS:
            self.initial_decay_value = self.DEFAULT_INITIAL_DECAY
        if self.half_decay_distance is None and self.kernel in self.WEIGHT_KERNELS:
            self.half_decay_distance = self.DEFAULT_HALF_DECAY
        if self.C is None:
            self.validate_kernel()
            self.C = self.DEFAULT_C[self.kernel]

