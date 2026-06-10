import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import click
import yaml

@click.command()
@click.option(
    "--config",
    default="configs/rag_qwen.yaml",
    help="Path to config"
)
def main(config):

    with open(config) as f:
        cfg = yaml.safe_load(f)

    print(cfg["model_name"])

    #загрузка данных
    articles = pd.read_json('data/articles.json')

    questions = pd.read_json('data/questions.json')

    ground_truth = pd.read_json('data/ground_truth.json')

    # загрузка модели
    MODEL_NAME = cfg["model_name"]
    
    model_alias = "qwen_0_5b"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to("cuda")

    def generate_answer(question):

        messages = [
            {
                "role": "user",
                "content": question
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.0,
                do_sample=False
            )

        answer = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return answer

    answers = []

    for question in questions["question"]:
        answers.append(
            generate_answer(question)
        )

    # сохранение результатов
    results_df = questions.copy()

    results_df[f"prediction_{model_alias}"] = answers

    results_df["ground_truth"] = ground_truth["answer"]

    results_df.to_csv(f'results/predict_{model_alias}_zero.csv', index=False)

if __name__ == "__main__":
    main()