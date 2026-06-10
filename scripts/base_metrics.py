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

    print(cfg["model_name"])

    DATA = 'rag_full.csv'

    predict_full = pd.read_csv(f'results/{DATA}')

    # Базовые метрики
    rouge = evaluate.load("rouge")

    rouge_results_rag_qwen = rouge.compute(
        predictions=predict_full['rag_qwen'],
        references=predict_full["ground_truth"].tolist()
    )

    rouge_results_rag_ministral = rouge.compute(
        predictions=predict_full['rag_ministral'],
        references=predict_full["ground_truth"].tolist()
    )

    bleu = evaluate.load("bleu")

    bleu_result_rag_qwen = bleu.compute(
        predictions=predict_full['rag_qwen'],
        references=[[x] for x in predict_full["ground_truth"].tolist()]
    )

    bleu_result_rag_ministral = bleu.compute(
        predictions=predict_full['rag_ministral'],
        references=[[x] for x in predict_full["ground_truth"].tolist()]
    )

    print('rouge_rag_qwen:', rouge_results_rag_qwen)
    print('rouge_rag_ministral:', rouge_results_rag_ministral)
    print('bleu_rag_qwen', bleu_result_rag_qwen)
    print('bleu_rag_ministral', bleu_result_rag_ministral)

if __name__ == "__main__":
    main()