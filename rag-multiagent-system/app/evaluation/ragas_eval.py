# RAGAS evaluation - measures faithfulness, answer relevancy, context precision
# Note: RAGAS uses an LLM internally so make sure GOOGLE_API_KEY (or OPENAI_API_KEY) is set
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def evaluate_ragas(questions, answers, contexts, ground_truths=None):
    # questions: list[str]
    # answers: list[str]   (generated)
    # contexts: list[list[str]]
    # ground_truths: optional list[str]
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    ds = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision]
    result = evaluate(ds, metrics=metrics)
    return result


if __name__ == "__main__":
    # Quick demo
    questions = ["What are Tesla's main risks?"]
    answers = ["Tesla's main risks include supply chain issues and competition."]
    contexts = [["Tesla faces risks from supply chain disruptions and rising competition in EVs."]]
    print(evaluate_ragas(questions, answers, contexts))
