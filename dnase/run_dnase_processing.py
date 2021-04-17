import pandas as pd
from prefect.executors import LocalDaskExecutor

from .process_dnase_flow import create_dnase_process_flow

WORKERS_NUM = 20
THREADS_PER_WORKER = 1
ROOT_DIR = "/mnt/NAS/home/abramov/"
EXPERIMENT_TYPE = "dnase"
ALIGNS_PATH = "/home/penzard/deepbeer/chosen.tsv"
RUN_DIR = "processed_dnase"


aligns = pd.read_table(ALIGNS_PATH)
align_names = [x for x in aligns["ALIGNS"]]

executor = LocalDaskExecutor(
    scheduler="threads",
    num_workers=WORKERS_NUM,
    threads_per_worker=THREADS_PER_WORKER,
    processes=False,
)

flow = create_dnase_process_flow()
state = flow.run(
    executor=executor,
    align_name=align_names,
    exp_type=[EXPERIMENT_TYPE] * len(align_names),
    root_dir=ROOT_DIR,
    run_dir=RUN_DIR,
)
