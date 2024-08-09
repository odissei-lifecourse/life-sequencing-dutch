# life-sequencing-dutch

Main code for the project [Modeling life outcomes](https://research-software-directory.org/projects/modeling-life-outcomes).



### For developers


#### Linting and code quality checks with `pre-commit`

Before a commit is stored (with `git commit -m "some message"`), `pre-commit` runs, on all the files staged for a commit, as specified in `.pre-commit-config.yaml` . `pre-commit` will
- automatically fix certain things, such as line endings and import ordering. If this happens, it will be necessary to re-stage the files for committing.
- complain about other errors that cannot be fixed automatically. for instance
  ```bash
  src/evaluation/report_utils.py:1528:5: S101 Use of `assert` detected
  ```
  - what `S101` means can be found in the ruff docs: https://docs.astral.sh/ruff/rules/assert/

Sometimes it makes sense to turn off checks for a specific line. This can be done with

```python
assert not_recommended # noqa: S101
```
