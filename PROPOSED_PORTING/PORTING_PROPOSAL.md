Porting proposal — automated Python 2 -> 3 repair (proposal)

Goal
----
Create reviewable patch files that perform a conservative, automated port of common Python-2 idioms to Python-3 across the repo. The patches are generated but not applied; you can inspect and apply them when ready.

What will be changed (conservative list)
--------------------------------------
- Replace `import cPickle` / `import cPickle as pickle` with a small shim that tries `cPickle` and falls back to `pickle`.
- Replace `xrange(` -> `range(`.
- Replace `.iteritems()` -> `.items()` (note: where code expects a list, the patch will replace with `list(<expr>.items())`).
- Replace `raw_input(` -> `input(`.
- (Optional) Convert simple `print` statements to `print(...)` — not included by default in this pass to reduce risk. We can add an option.

Scope and safety
----------------
- The generator script will only write patch files under `PROPOSED_PORTING/patches/` and will not modify repository files.
- Patch format: unified diff (.patch) suitable for `git apply` or `patch -p0`.
- The script makes conservative changes using regex heuristics. It may need manual review for complex cases (e.g., custom `cPickle` variable names, uses of `xrange` in string contexts, or `.iteritems()` where the code expects an iterator vs list).

How to review and apply
-----------------------
1. Inspect patches:
   - Open `PROPOSED_PORTING/patches/` and review the `.patch` files.
2. Create a branch and apply patches:
   - git checkout -b port/python3-proposed
   - git apply --index PROPOSED_PORTING/patches/*.patch
3. Run tests / smoke imports; fix issues manually if needed.

Notes
-----
- Interactive annotation utilities use `raw_input` intentionally. If you prefer we can exclude directories (for example `corpus/`) from the automatic replacement. The generator supports an exclusion list.
- After applying patches I recommend running a small smoke test: `python -c "import deep_disfluency.evaluation.disf_evaluation; print('ok')"` and then run the `EACL_2017.ipynb` notebook cells.

Files created by this proposal
-----------------------------
- `PROPOSED_PORTING/PORTING_PROPOSAL.md` (this file)
- `PROPOSED_PORTING/make_porting_patches.py` — generates .patch files.
- `PROPOSED_PORTING/README_APPLY.md` — quick apply instructions (also included below).

If this looks good I created a script that will generate the patches for review. Run it with your Python 3 virtualenv active.
