import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import click
import yaml

@click.command()
@click.option(
    "--config",
    default="configs/rag.yaml",
    help="Path to config"
)
def main(config):

    with open(config) as f:
        cfg = yaml.safe_load(f)

    print(cfg["model_name"])

    # Загрузка данных
    articles = pd.read_json('data/articles.json')

    questions = pd.read_json('data/questions.json')

    ground_truth = pd.read_json('data/ground_truth.json')

    articles_df = articles.copy()

    # объединим тексты и тему
    articles_df['full'] = (articles["title"] +"\n\n" +articles["text"]).tolist()

    # Формируем эмбеддинги запроса и текста
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    texts = articles_df['full'].tolist()

    vectors = embeddings.embed_documents(texts) # текст в эмбеддинги

    def retrival(query):

        query_vector = embeddings.embed_query(query)

        scores = cosine_similarity([query_vector], vectors) # косинусное сходство запрос - тексты

        best_idx = np.argmax(scores)

        answer = articles_df.iloc[best_idx]["text"] # находим ответ

        return answer

    # Загрузка модели и промта
    MODEL_NAME = cfg["model_name"]
    
    model_alias = "qwen_0_5b"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to("cuda")

    def generate_answer(question):
        
        context = retrival(question)
        messages = [
            {
                "role": "system",
                "content": "Ты ассистент, который отвечает только по предоставленному контексту. Если ответа нет в контексте, скажи: 'В контексте нет информации.'"
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос:\n{question}"
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

    # Сохранение результатов
    results_df = questions.copy()

    results_df[f"rag_{model_alias}"] = answers

    results_df["ground_truth"] = ground_truth["answer"]

    results_df.to_csv(f'results/rag_{model_alias}.csv', index=False)

if __name__ == "__main__":
    main()
