from promptx_core.metric.rouge import Rouge


def test_rouge_single_reference():
    rouge = Rouge()
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]

    result = rouge.batch(references, predictions, dspy_example=False)

    assert isinstance(result, float)
    assert 0 <= result <= 1.0
    # assert set(results.keys()) == {"rouge1", "rouge2", "rougeL", "rougeLsum"}


def test_rouge_multi_reference():
    rouge = Rouge()
    predictions = ["hello there", "general kenobi"]
    references = [["hi there", "hello there"], ["general kenobi", "hello general"]]

    result = rouge.batch(references, predictions, dspy_example=False)

    assert isinstance(result, float)
    assert 0 <= result <= 1.0
    # assert set(results.keys()) == {"rouge1", "rouge2", "rougeL", "rougeLsum"}


def test_rouge_no_aggregator():
    rouge = Rouge()
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]

    result = rouge.batch(
        references, predictions, dspy_example=False, use_aggregator=False
    )

    assert isinstance(result, float)
    assert 0 <= result <= 1.0
    # assert set(results.keys()) == {"rouge1", "rouge2", "rougeL", "rougeLsum"}


def test_rouge_custom_types():
    rouge = Rouge(rouge_type="rougeLsum")
    predictions = ["hello there"]
    references = ["hello there"]

    result = rouge.batch(references, predictions, dspy_example=False)

    assert isinstance(result, float)
    assert 0 <= result <= 1.0
    assert result == 1.0  # Perfect match for rouge2
