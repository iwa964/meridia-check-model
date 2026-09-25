# meridia-check-model

Meridia 的掷骰检定决策训练项目。此仓库是训练数据的后续维护位置。

This repository is the canonical home for Meridia dice-check decision training data.

## Contribution workflow

所有改动必须在独立分支完成并通过 PR 提交，禁止直接写入 `main`。当前训练数据标注在 `codex/dice-rolling-training` 分支继续维护。

Make all changes on a dedicated branch and submit them through a pull request. Do not commit directly to `main`. Continue the current annotation session on `codex/dice-rolling-training`.

## Training dataset

- File: [`training_files/dice_rolling_train.json`](training_files/dice_rolling_train.json)
- Workflow: generate a general scene, record the creator's label, save it, then present the next pending scene.
- Preserve the creator's exact response and all existing annotations, corrections, conditional branches, alternatives, parameterized checks, and skipped scenes.
- A skill plus a difficulty defaults to a unidirectional check. Validate skill and attribute names against the catalogs referenced in the dataset; ask the creator to resolve unknown names.
- General scenarios are for training; game-specific scenarios are reserved for testing. Game-specific exceptions already recorded in the dataset retain their export restrictions.
- This is an annotation dataset. Model training and runtime evaluation are not implemented by this import; unresolved parameters and runtime inputs still require preparation.

## Migration

This pull request imports the dataset without content changes from [`iwa964/MeridiaGame`](https://github.com/iwa964/MeridiaGame/blob/f81958dd6cc72a62e36c79d8e367e850eee40d73/training_files/dice_rolling_train.json), commit `f81958dd6cc72a62e36c79d8e367e850eee40d73`.

Continue dataset updates here. References to MeridiaGame inside the JSON identify the original skill/attribute catalogs and rule definitions; they remain unchanged as provenance.
