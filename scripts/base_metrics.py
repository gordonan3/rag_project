import pandas as pd
import evaluate
import click
import yaml

@click.command()
@click.option(
    "--config",
    default="configs/metrics.yaml",
    help="Path to config"
)
def main(config):

    with open(config) as f:
        cfg = yaml.safe_load(f)

    input_file = cfg["input_file"]
    ground_truth_col = cfg["ground_truth_column"]
    prediction_cols = cfg["prediction_columns"]

    predict_full = pd.read_csv(input_file)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    all_metrics = {}

    for pred_col in prediction_cols:
        print(f"\n=== Metrics for {pred_col} ===")

        references = predict_full[ground_truth_col].astype(str).tolist()
        predictions = predict_full[pred_col].astype(str).tolist()

        rouge_result = rouge.compute(
            predictions=predictions,
            references=references
        )

        bleu_result = bleu.compute(
            predictions=predictions,
            references=[[x] for x in references]
        )

        all_metrics[pred_col] = {
            "rouge": rouge_result,
            "bleu": bleu_result,
        }

        print("ROUGE:", rouge_result)
        print("BLEU:", bleu_result)

    if cfg.get("metrics", {}).get("comet", False):
        from comet import download_model, load_from_checkpoint

        comet_cfg = cfg["comet"]

        model_path = download_model(comet_cfg["model_name"])
        comet_model = load_from_checkpoint(model_path)

        for pred_col in prediction_cols:
            print(f"\n=== COMET for {pred_col} ===")

            data = []

            for _, row in predict_full.iterrows():
                data.append({
                    "src": str(row["question"]),
                    "mt": str(row[pred_col]),
                    "ref": str(row[ground_truth_col]),
                })

            comet_output = comet_model.predict(
                data,
                batch_size=comet_cfg.get("batch_size", 8),
                gpus=comet_cfg.get("gpus", 0)
            )

            comet_mean = float(sum(comet_output.scores) / len(comet_output.scores))

            all_metrics[pred_col]["comet"] = {
                "mean": comet_mean,
                "scores": comet_output.scores,
            }

            print("COMET mean:", comet_mean)

    print("\n=== ALL METRICS ===")
    print(all_metrics)


if __name__ == "__main__":
    main()