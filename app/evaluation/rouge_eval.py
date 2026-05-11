# ROUGE evaluation - compares generated answer with reference answer
from rouge_score import rouge_scorer


def evaluate_rouge(generated, reference):
    # Returns ROUGE-1, ROUGE-2 and ROUGE-L scores
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def evaluate_batch(generated_list, reference_list):
    # Evaluate a list of generated vs reference answers and average results
    assert len(generated_list) == len(reference_list)

    totals = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for g, r in zip(generated_list, reference_list):
        scores = evaluate_rouge(g, r)
        for k in totals:
            totals[k] += scores[k]

    n = len(generated_list)
    return {k: v / n for k, v in totals.items()}


if __name__ == "__main__":
    # Quick demo
    gen = "Tesla faced production challenges in 2024 due to supply chain issues."
    ref = "In 2024, Tesla struggled with production because of supply chain problems."
    print(evaluate_rouge(gen, ref))
