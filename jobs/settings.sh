# =====================================================================
# Shared settings for the SLURM job scripts. EDIT THIS FILE, not the scripts.
# =====================================================================

# --- where things live ---
DATA_DIR=/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/L1NNTau_data          # input ntuples
MODEL_DIR=/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/models/$USER/study1   # trained models go here (one study = one folder)
PLOT_DIR=/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/plots/$USER/study1     # plots go here
LOG_DIR=$REPO_DIR/logs                                                          # job logs

# --- software ---
ENV_NAME=tau_nn                 # conda/mamba environment (see README "Installation")

# --- SLURM resources for each training job ---
PARTITION=gpu                   # gpu_test for quick tests (short time limit)
GPUS=1
CPUS=4
MEM=32G
TIME=0-04:00                    # days-hours:minutes
MAX_PARALLEL=10                 # at most this many jobs of one list run at the same time
