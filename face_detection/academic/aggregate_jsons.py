import argparse
import itertools
import json
import multiprocessing as mp

from pathlib import Path


def process_model_on_dataset(
    args: argparse.Namespace, model: str, dataset: str
) -> None:
    print(("=" * 40) + f" Aggregating JSONs for <{model}> on <{dataset}> " + ("=" * 40))

    model_dataset_file = args.results_root / f"{model}_{dataset}.json"
    aggregated_preds = []

    for i, preds_file in enumerate(
        (args.results_root / model / dataset).glob("**/*.json"), start=1
    ):
        preds_path = str(preds_file).replace(str(args.results_root), "").lstrip("/")

        with open(preds_file, "r") as f:
            response = json.load(f)

        aggregated_preds.append({"json": preds_path, "response": response})
        if args.verbose:
            print(f"{model} <{dataset}> [{i}]")

    with open(model_dataset_file, "w") as f:
        json.dump(aggregated_preds, f)

    print(
        ("=" * 40)
        + f" Finished aggregating JSONs for <{model}> on <{dataset}> "
        + ("=" * 40)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregates all of the JSON prediction files into one JSON per model, dataset pair"
    )
    parser.add_argument(
        "results_root",
        type=Path,
        help="Where the results root is placed after predictions are made",
    )
    parser.add_argument(
        "--models",
        default=None,
        type=str,
        nargs="+",
        help="Which models made predictions. By default will be MogFace, TinaFace, and YOLOv5-Face.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        type=str,
        nargs="+",
        help="Which datasets the models made predictions on. By default will be Adience, MIAP, and UTKFACE.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print progress"
    )
    args = parser.parse_args()

    args.models = args.models or ["mogface", "tinaface", "yolov5"]
    args.datasets = args.datasets or ["adience", "miap", "utkface"]

    if len(list(itertools.product(args.models, args.datasets))) > 1:
        processes = [
            mp.Process(target=process_model_on_dataset, args=(args, model, dataset))
            for model, dataset in itertools.product(args.models, args.datasets)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
    else:
        process_model_on_dataset(args, args.models[0], args.datasets[0])

    print("Finished aggregating JSONs")
