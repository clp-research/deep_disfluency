How to review and apply the proposed porting patches

1. Inspect the generated patches (they do not modify files):

   python3 PROPOSED_PORTING/make_porting_patches.py

   This will generate patch files under `PROPOSED_PORTING/patches/` mirroring the
   repository path structure. Open those `.patch` files and review changes.

2. Create a branch and apply patches:

   git checkout -b port/python3-proposed
   git apply --index PROPOSED_PORTING/patches/**/*.patch

   If any patch fails, inspect the `.patch` file and apply manually or fix conflicts.

3. Run quick smoke imports and tests:

   # Activate your Python 3 venv first
   python -c "import deep_disfluency.evaluation.disf_evaluation; print('import ok')"

   Then open `EACL_2017.ipynb` and run the first cells to verify imports and basic usage.

4. If all good, commit and push:

   git add -A
   git commit -m "Apply automated Python2->3 porting patches (reviewed)"
   git push origin port/python3-proposed

Notes
-----
- The generator is conservative but not perfect. Please review patches before applying.
- The script currently replaces `raw_input` -> `input` across the repo; if you want to skip interactive corpus annotation tools, edit `EXCLUDE_DIRS` in `make_porting_patches.py`.
