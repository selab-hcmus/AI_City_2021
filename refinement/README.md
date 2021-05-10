# Refinement Module

This module handles the following task:
- Refine the result obtained from the [`COOT retrieval model`](../retrieval_model).

## Module organization
- `data`: contains necessary files used for refinement.
- `results`: contains the final result to for this retrieval system.
- `notebook`: contains fundamental steps to test/run this module.

## Prepare
This module needs previous steps' results:
- [Classifier](../classifier): Visual label prediction of vehicle type and color.
- [Detector](../detector): Visual label prediction of stop and turn event.
- [SRL Extraction](../srl_extraction): Text label prediction of vehicle type, color and motion event.
- [Retrieval Model](../retrieval_model): Result obtained from the COOT model.

## Run
```
cd ./refinement
python main.py
```
Result will be saved to `refinement/results`
