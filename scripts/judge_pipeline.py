import os
import pandas as pd
import numpy as np
import re

from tqdm.auto import tqdm

from groq import Groq, APIConnectionError, RateLimitError, APIStatusError
from dotenv import load_dotenv
import json
import time
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

    load_dotenv()

    JUDGE_MODEL = "llama-3.1-8b-instant"
    DATA = "predict_full.csv"

    predict_full = pd.read_csv(f'results/{DATA}')

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    def build_judge_prompt_rag(question, ground_truth, prediction):
        return f"""
    Ты строгий эксперт-оценщик ответов на вопросы по корпоративной базе знаний.

    Оцени ответ модели относительно эталонного ответа.

    Критерии:
    1. factual_correctness — фактическая корректность.
    2. completeness — полнота ответа.
    3. relevance — отвечает ли ответ на заданный вопрос.
    4. hallucination — есть ли выдуманная или неподтвержденная информация (обратная шкала)

    Поставь score от 0 до 5:
    0 — полностью неверно
    1 — почти неверно
    2 — частично верно, но много ошибок
    3 — в целом верно, но неполно
    4 — почти полностью верно
    5 — полностью верно и достаточно полно

    Для hallucination в обратном порядке (0 — много галлюцинаций, 5 — нет галлюцинаций)

    Верни только JSON без markdown.

    Вопрос:
    {question}

    Эталонный ответ:
    {ground_truth}

    Ответ модели:
    {prediction}

    JSON формат:
    {{
    "score": 0,
    "factual_correctness": 0,
    "completeness": 0,
    "relevance": 0,
    "hallucination": 0,
    "comment": "краткое объяснение"
    }}
    """

    def parse_judge_answer(text):
        text = text.strip()

        # убираем markdown fences
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        # вытаскиваем JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            text = match.group(0)

        return json.loads(text)

    def judge_answer(row, prediction_col, max_retries=5):
        prompt = build_judge_prompt_rag(
            question=row["question"],
            ground_truth=row["ground_truth"],
            prediction=row[prediction_col]
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=300,
                )

                text = response.choices[0].message.content.strip()

                try:
                    result = parse_judge_answer(text)
                    result["raw_response"] = text
                    result["prediction_col"] = prediction_col
                    return result

                except json.JSONDecodeError:
                    return {
                        "score": None,
                        "factual_correctness": None,
                        "completeness": None,
                        "relevance": None,
                        "hallucination": None,
                        "comment": "JSON parse error",
                        "raw_response": text,
                        "prediction_col": prediction_col,
                    }

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                wait = 2 ** attempt
                print(f"Groq error: {type(e).__name__}. Retry in {wait}s...")
                time.sleep(wait)

        return {
            "score": None,
            "factual_correctness": None,
            "completeness": None,
            "relevance": None,
            "hallucination": None,
            "comment": "Groq API failed after retries",
            "raw_response": None,
            "prediction_col": prediction_col,
        }


    judge_df = predict_full.copy()

    results_model = []

    for _, row in tqdm(judge_df.iterrows(), total=len(judge_df)):
        result_model = judge_answer(row, "rag_qwen")
        results_model.append(result_model)
        time.sleep(1.5)

    judge_qwen_df = pd.DataFrame(results_model)

    judge = judge_qwen_df.mean(numeric_only=True)

    print(judge)

    judge_qwen_df.to_csv('results/judge_result.csv', index=False)

if __name__ == "__main__":
    main()