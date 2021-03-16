import sys
sys.path.append("/home/penzard/deepbeer/deepbeer_git/dnase")
import pandas as pd

from prefect.executors import LocalDaskExecutor

from process_dnase_flow import create_dnase_process_flow


WORKERS_NUM = 20
THREADS_PER_WORKER = 1
ROOT_DIR = "/mnt/NAS/home/abramov/"
EXPERIMENT_TYPE = "dnase"
ALIGNS_PATH = "/home/penzard/deepbeer/chosen.tsv"


aligns = pd.read_table(ALIGNS_PATH)
align_names = [x for x in aligns['ALIGNS']]

executor = LocalDaskExecutor(scheduler="threads", 
                              num_workers=WORKERS_NUM, 
                              threads_per_worker=THREADS_PER_WORKER,
                              processes=False)

flow = create_dnase_process_flow()
state = flow.run(executor=executor,
                 align_name=align_names,
                 exp_type=["dnase"] * len(align_names),
                 root_dir= "/mnt/NAS/home/abramov/",
                 run_dir="processed_dnase")
