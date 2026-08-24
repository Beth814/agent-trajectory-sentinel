# What is in the healthy null

**SUPPORTING / diagnostic evidence.** Nothing here relabels, admits or rejects an episode. It reports the composition of the healthy populations every per-deployment threshold is derived from, by re-running each task's own `success_fn` over the committed traces.

Healthy admission in these corpora is outcome-independent (`collect_framework_real_traces.check_admission`): a healthy run is admitted for existing, never for scoring well. That is the right rule -- it cannot silently drop difficult-but-valid behaviour -- but it also never consults the verifier, so the corpus records `success = None` for every healthy episode and says nothing about whether the agent finished the task.

## langgraph7b_real2

- healthy attempted: 120; admitted **120**; rejected **0**
- injected rejected at admission (for contrast): 2
- admission reason, all healthy episodes: `healthy, outcome-independent admission`
- task mix: arxiv_paper_search 60, multi_city_weather 60
- model: qwen2.5:7b
- trajectory length: min 4, median 7, max 10
- **verifier re-run offline: 117 pass, 3 fail, 0 unverifiable**
- episodes too short to produce any score (T <= washout 3): **0**

Rejection reasons recorded at collection:
- injection never applied (no-op positive): 2

## autogen7b_real2

- healthy attempted: 120; admitted **120**; rejected **0**
- injected rejected at admission (for contrast): 24
- admission reason, all healthy episodes: `healthy, outcome-independent admission`
- task mix: arxiv_paper_search 60, multi_city_weather 60
- model: qwen2.5:7b
- trajectory length: min 2, median 5, max 12
- **verifier re-run offline: 102 pass, 18 fail, 0 unverifiable**
- episodes too short to produce any score (T <= washout 3): **21**

Rejection reasons recorded at collection:
- injection never applied (no-op positive): 24

## Trajectory length against verified success

| corpus | verifier | n | T min | T median | T max | too short to score |
|---|---|---|---|---|---|---|
| langgraph7b_real2 | pass | 117 | 4 | 7 | 10 | 0 |
| langgraph7b_real2 | fail | 3 | 4 | 6 | 6 | 0 |
| autogen7b_real2 | pass | 102 | 3 | 7 | 12 | 6 |
| autogen7b_real2 | fail | 18 | 2 | 2 | 5 | 15 |
