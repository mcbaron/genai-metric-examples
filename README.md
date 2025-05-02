# genai-metric-examples

A sketch repository that illustrates how to evaluate and optimize LLM system prompts using classic and modern text evaluation metrics (BLEU, METEOR, ROUGE, token counts, simple semantic overlap, and DSPy-derived semantic F1). It also outlines a structure for comparing multiple LLMs and for experimenting with prompt optimization strategies (e.g., MIPROv2, Bootstrap Few-Shot with Optuna/Random Search).

This is not a full working PoC. It is meant to demonstrate an understanding of how evaluation metrics apply to LLM prompt evaluation and optimization, and how an optimization loop could be configured.

## Why this exists

- **Prompt quality matters**: Small changes in a system prompt can significantly affect LLM behavior and downstream outcomes.
- **Metrics provide signal**: Text similarity metrics like **BLEU**, **METEOR**, and **ROUGE** help quantify output quality against references; token-based metrics estimate cost/latency; lightweight semantic checks provide fast heuristics; DSPy-based metrics hint at more semantic evaluation.
- **Comparable experiments**: A shared evaluation suite enables comparing different prompts and model backends more fairly.

## What’s in this sketch

- **Metrics (`metric/`)**
  - `Bleu` (SacreBLEU) — corpus and sentence scoring, multi-reference support.
  - `Meteor` (NLTK) — unigram precision/recall with fragmentation penalty, multi-reference support.
  - `Rouge` (rouge-score) — ROUGE-1/2/L/Lsum with bootstrap aggregation.
  - `TokensMetric` (tiktoken/Anthropic) — estimate tokens for cost/latency.
  - Simple heuristics: `SemanticMetric` (token overlap Jaccard), `LengthMetric`, `HasContextMetric`, `HasInstructionMetric`.
  - `DspySemanticF1Metric` — illustrates semantic evaluation via DSPy components.
  - `OptMetric` — weighted aggregation over multiple metrics to form a single optimization target.
- **Configs (`config/`)**
  - `ModelConfig` — model list/selection, generation params, default metric suite.
  - Optimization configs for sketch strategies: `MIPROv2`, `BootstrapFewShotWithOptuna`, `BootstrapFewShotWithRandomSearch` with compile settings and thresholds.
  - `Config` — top-level container combining model, optimization, optional retriever, and custom payloads.
- **Tests (`tests/`)**
  - Unit tests for `Bleu`, `Meteor`, and `Rouge` usage patterns.
  - E2E YAML examples for optimization runs (used as illustrative configs in this sketch).

## High-level architecture

- Metrics implement `BaseMetric` / `BaseReferenceMetric` interfaces for consistent `forward`, `batch`, and error handling semantics.
- `OptMetric` composes weighted scores from multiple metrics, allowing a single scalar objective for search.
- Config models (Pydantic) express model choices, metrics, and optimization parameters and can be serialized to/from YAML/JSON.
- Example E2E configs show how you might wire model settings and optimization strategies together.

## Metric primers (intuition and applicability)

- **BLEU**: n-gram precision with brevity penalty. Good for machine translation–style overlap. Less sensitive to synonyms/semantics. Use for conservative overlap checks and regressions.
- **METEOR**: unigram precision/recall with stemming/synonyms and fragmentation penalty. Typically correlates better with human judgment than BLEU on some tasks. Useful when word choice variants should get credit.
- **ROUGE**: recall-oriented (variants: 1/2/L/Lsum). Common for summarization. Good when you care about coverage of reference content.
- **Token count**: proxy for cost and latency; helpful constraint/regularizer during optimization.
- **DSPy semantic F1 (sketch)**: aims to capture semantic correctness via precision/recall over decomposed facts. Illustrative here; real use requires careful setup.
- **Simple heuristics**: length/context/instruction presence can guide early pruning or guardrails for system prompts.

## Comparing models

- `config/utils.get_default_model_names()` lists example target backends (Anthropic Claude variants as placeholders).
- `ModelConfig` holds a primary `model_name` and a list of `model_names` to signal a space for comparison; you could iterate the same evaluation suite across candidates and report per-model metric tables.

## Example: using metrics directly

```python
from promptx_core.metric import Bleu, Meteor, Rouge, TokensMetric

predictions = ["hello there general kenobi"]
references = [["hello there general kenobi", "hello there !"]]

bleu = Bleu()
meteor = Meteor()
rouge = Rouge(rouge_type="rouge1")
tokens = TokensMetric()

# Single
b = bleu(references[0], predictions[0], dspy_example=False)
m = meteor(references[0], predictions[0], dspy_example=False)
r = rouge(references[0], predictions[0], dspy_example=False)

# Batch
b_avg = bleu.batch(references, predictions, dspy_example=False)

# Tokens (OpenAI tokenizer family)
num_tokens = tokens("Your system prompt or output text")
```

## Example: aggregating metrics for optimization

```python
from promptx_core.metric import OptMetric, Meteor, Rouge, Bleu

metric = OptMetric(metrics_weights={
    Meteor(): 0.5,   # maximize
    Rouge("rougeL"): 0.3,  # maximize
    Bleu(): 0.2,     # maximize
})

score = metric(
    reference="gold standard answer",
    pred="model output",
    dspy_example=False,
)
```

## Configuration examples (sketch)

See `tests/e2e/*.yaml` for illustrative configs. For example:

```yaml
prompt:
  model:
    temperature: 0.7
    model_names:
      - anthropic.claude-v3-5-sonnet
    top_p: 0.9
    metrics:
      - "bleu"
  optimization:
    optimizer_type: BootstrapFewShotWithOptuna
    params:
      max_bootstrapped_demos: 8
      max_labeled_demos: 16
      num_candidates: 3
      metric_threshold: 0.85
      num_threads: 4
data:
  test_size: 1
```

These demonstrate how you could vary optimizer type and thresholds. In a full system, such a config would drive an optimization loop that proposes candidate prompts, evaluates them via the metric suite, and selects/improves prompts iteratively.

## Installation (sketch)

This repo depends on external packages for metrics:
- sacrebleu>=1.4.12
- nltk (with wordnet, punkt/punkt_tab, omw-1.4 datasets)
- rouge-score
- tiktoken
- dspy (optional, for DSPy semantic metric)
- anthropic (optional, for Claude token counting)
- pydantic, pyyaml

Example:
```bash
pip install sacrebleu>=1.4.12 nltk rouge-score tiktoken pydantic pyyaml
# optional extras
pip install dspy-ai anthropic
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

## Limitations and notes

- This is a conceptual sketch. The optimization loop and data plumbing referenced in tests point to an external core (`promptx_core`) and are illustrative only.
- Some metrics assume simple string inputs and do not include advanced preprocessing/tokenization beyond library defaults.
- Semantic scoring beyond surface overlap (e.g., factuality, faithfulness) requires more sophisticated evaluators and task-specific prompts.
- Token counting varies by provider and model; `TokensMetric.calculate_claude` demonstrates the Anthropic API pattern but requires credentials and network access.

## Roadmap ideas (if turning into a PoC)

- Wrap a runnable CLI to load a config, evaluate candidate prompts across multiple LLMs, and export metric dashboards.
- Add human-in-the-loop labeling and pairwise preference aggregation.
- Incorporate modern evaluators (e.g., LLM-as-judge, factuality checks, safety, toxicity) and calibration.
- Provide standardized datasets and task suites with task-specific references and instructions.
- Add statistical testing and confidence intervals across runs and seeds.
