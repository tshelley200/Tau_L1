#!/bin/bash
# Submit one SLURM job per line of a job list (as a job array).
#
#   jobs/submit_train.sh                       # uses jobs/train_jobs.txt
#   jobs/submit_train.sh jobs/my_jobs.txt      # another list
#   DRY_RUN=1 jobs/submit_train.sh             # only print the commands, submit nothing
#
# Resources (partition, GPU, memory, time) and folders come from jobs/settings.sh.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_DIR/jobs/settings.sh"

JOB_LIST=$(realpath "${1:-$REPO_DIR/jobs/train_jobs.txt}")
[ -f "$JOB_LIST" ] || { echo "Job list not found: $JOB_LIST"; exit 1; }

LINES=$(sed 's/#.*//' "$JOB_LIST" | awk 'NF')
N=$(echo "$LINES" | awk 'NF' | wc -l)
[ "$N" -gt 0 ] || { echo "No jobs in $JOB_LIST"; exit 1; }

# names must be unique (they become the model file names)
DUP=$(echo "$LINES" | awk '{print $1}' | sort | uniq -d)
[ -z "$DUP" ] || { echo "Duplicate job names in $JOB_LIST: $DUP"; exit 1; }

echo "$N job(s) from $JOB_LIST  ->  models in $MODEL_DIR"
echo "$LINES" | awk '{printf "  %2d  %s\n", NR, $1}'

if [ "${DRY_RUN:-0}" = "1" ]; then
    for i in $(seq 1 "$N"); do
        REPO_DIR=$REPO_DIR JOB_LIST=$JOB_LIST SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash "$REPO_DIR/jobs/train_array.sbatch" | tail -1
    done
    exit 0
fi

mkdir -p "$LOG_DIR" "$MODEL_DIR"
sbatch --array=1-"$N"%"$MAX_PARALLEL" \
       --partition="$PARTITION" --gres=gpu:"$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
       --output="$LOG_DIR/%x_%A_%a.out" \
       --export=ALL,REPO_DIR="$REPO_DIR",JOB_LIST="$JOB_LIST" \
       "$REPO_DIR/jobs/train_array.sbatch"
echo "Logs: $LOG_DIR/tau_train_<jobid>_<line>.out   |   check status: squeue -u \$USER"
