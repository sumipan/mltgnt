# Testing policy

Tests protect contracts, not implementations. A test may pin "this input
gives this output and these side effects". It must not pin function names,
private helpers, call order, source text, or documentation wording.

## Principles

1. Pin behaviour, not shape. Call the function and check the result.
2. One fact lives in one place. Do not copy constants or config into tests;
   derive expected values from the source of truth.
3. Name tests after behaviour. Issue numbers belong in docstrings only.
4. Fake only process boundaries: subprocesses, HTTP, clocks, the filesystem.
   Do not patch private helpers of the module under test and assert their
   call arguments.
5. Repository-wide invariants are convention tests under `tests/conventions/`,
   one per invariant.
6. Tests are a liability, not an asset. When a behaviour is removed, remove
   its tests in the same PR. Verify removals once in review; do not keep a
   permanent "X no longer exists" test.
7. Keep the volume in check: test lines should not exceed source lines, one
   contract per test, new files under 800 lines.

## Rules

| ID | Rule |
|----|------|
| T1 | Do not read source text or bytecode (`inspect.getsource`, `__code__.co_consts`, `ast.parse` of `src/`) outside `tests/conventions/`. |
| T2 | Do not assert the wording of `README.md`, `CHANGELOG.md`, or docs. Convention scans for forbidden patterns are fine. |
| T3 | Do not keep removal checks (`test_*_removed`, `not hasattr(...)`, `pytest.raises(ImportError)`) as permanent tests. |
| T4 | Do not name files or functions after issue numbers, acceptance-criteria numbers, or phases. |
| T5 | Do not assert call arguments of private helpers in the module under test. Verify boundaries, check results for the rest. |
| T6 | Do not snapshot structural counts or enumerations against directory scans. Public `__all__` snapshots are contracts and are allowed. |
| T7 | Do not import host-specific vocabulary (labels, template names, phase ids) into this repository's tests. |
| T8 | Keep test files under 1,000 lines; split by module when they grow. |
| T9 | Do not write characterization tests that freeze current behaviour before a change. Write one test per acceptance criterion. |
| T12 | Coverage is an observation. The `--cov-fail-under` threshold stays at its current value and is never raised; do not add tests only to satisfy it. |

## Removing tests

For each candidate, ask in order: what contract does it protect; does another
behaviour test already protect it; is the contract worth keeping. Delete when
in doubt; history is in git. Rewrite only when the contract matters and no
behaviour test exists, and only if the rewrite is not longer than the original.

The inventory tool lives in the host repository:
`python3 scripts/test-policy-audit.py --repo <path-to-this-repo>` from a
`sumipan/nexus` checkout lists candidates per rule.
