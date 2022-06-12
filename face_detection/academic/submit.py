import argparse
import subprocess

from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submits SLURM experiments for "
        '"Are Commercial Face Detection Models as Biased as Academic Models?"'
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices={"tinaface", "scrfd", "yolov5", "mogface"},
        help="The model to perform the experiments on.",
    )
    parser.add_argument(
        "--image_file",
        required=True,
        type=str,
        help="The path to the image file for the given dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["adience", "miap", "utkface"],
        help="Which dataset to evaluate on.",
    )
    parser.add_argument(
        "--time",
        required=True,
        type=str,
        help="Length of time needed for the experiment.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="default",
        choices=["cpu", "default", "high", "medium", "scavenger"],
        help="Which QOS partition to request.",
    )
    parser.add_argument(
        "--gpu",
        default="gpu:1",
        type=str,
        choices=[
            "gpu:rtx2080:1",
            "gpu:rtx2080ti:1",
            "gpu:rtxa4:1",
            "gpu:rtx30:1",
            "gpu:rtx3070:1",
        ],
        help="Which gpu allocation to use.",
    )
    args = parser.parse_args()

    # The absolute path to the containing directory
    current_dir = Path(__file__).resolve().parent

    # Make the model specific log/err directory
    (current_dir / "output" / args.model).mkdir(parents=True, exist_ok=True)

    with open(current_dir / "infer.sh", "r") as f:
        infer_script = f.read()

    experiment_script = infer_script.format(
        model=args.model,
        dataset=args.dataset,
        partition=args.partition,
        time=args.time,
        gpu=args.gpu,
    )

    # Write a temporary slurm script with experiment details
    with open(current_dir / "temp.sh", "w") as f:
        f.write(experiment_script)

    # Command used to submit the slurm script
    submit_command = f"sbatch --export=MODEL={args.model},IMAGE_FILE={args.image_file},DATASET={args.dataset} temp.sh"

    # Submitting the job
    exit_status = subprocess.call(submit_command, shell=True)

    # Job did not submit properly
    if exit_status != 0:
        print(f"Job {submit_command} failed to submit")

    # Remove the temporary slurm script
    (current_dir / "temp.sh").unlink()
