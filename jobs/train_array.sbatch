#!/bin/bash
#SBATCH -J tau_train
# One task of a job array = one line of the job list. Don't submit this file directly: use jobs/submit_train.sh
# (it sets the partition / GPU / memory / time from jobs/settings.sh and tells this script which list to read).

source "$REPO_DIR/jobs/settings.sh"

# --- pick line number $SLURM_ARRAY_TASK_ID of the list (comments and empty lines removed) ---
LINE=$(sed 's/#.*//' "$JOB_LIST" | awk 'NF' | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r NAME ARGS <<< "$LINE"
if [ -z "$NAME" ]; then echo "No line $SLURM_ARRAY_TASK_ID in $JOB_LIST"; exit 1; fi

CMD="python -u training.py --data_dir $DATA_DIR --model_out $MODEL_DIR/$NAME.h5 $ARGS"
echo "[$(date)] job $SLURM_ARRAY_TASK_ID: $NAME"
echo "$CMD"
if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi   # DRY_RUN=1: only print the command

# --- software environment ---
module load python
source activate "$ENV_NAME"
export TF_CPP_MIN_LOG_LEVEL=2   # hide TensorFlow's harmless info messages

cd "$REPO_DIR/merged_training"
$CMD
RC=$?
echo "[$(date)] done (exit code $RC)"
exit $RC   # so SLURM marks the job FAILED if training crashed
