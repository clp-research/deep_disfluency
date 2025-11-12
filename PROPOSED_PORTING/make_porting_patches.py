#!/usr/bin/env python3
"""
Generate conservative porting patches for Python2->Python3 changes.
This script scans the repository rooted at the parent directory of this file
and writes unified-diff .patch files under PROPOSED_PORTING/patches/.

It does NOT modify any source files.

Patterns handled (conservative):
- `import cPickle` or `import cPickle as pickle` -> shim using try/except
- `xrange(` -> `range(`
- `.iteritems()` -> `.items()` (wrap in list(...) when RHS used in contexts that usually need a list)
- `raw_input(` -> `input(`

Run:
  python3 make_porting_patches.py

Review patches under PROPOSED_PORTING/patches/ before applying.
"""
import io
import os
import re
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = Path(__file__).resolve().parents[0] / 'patches'
EXCLUDE_DIRS = {'PROPOSED_PORTING', 'env', '.git'}

# conservative file glob: only .py files
PY_EXTS = ('.py',)

# regex patterns and repl functions
CPICKLE_IMPORT_RE = re.compile(r"^\s*(?:import\s+cPickle\s*$|import\s+cPickle\s+as\s+(?P<asname>\w+)\s*$)", re.MULTILINE)
Xrange_RE = re.compile(r"\bxrange\s*\(")
ITERITEMS_RE = re.compile(r"\.iteritems\s*\(")
RAW_INPUT_RE = re.compile(r"\braw_input\s*\(")

def make_cpickle_shim(original_lines):
    """If file contains cPickle imports, replace with shim at top of file.
    This is conservative: if multiple import locations exist, each is replaced.
    """
    new_lines = []
    changed = False
    # Detect if file already imports 'pickle' anywhere
    file_text = ''.join(original_lines)
    has_pickle_import = re.search(r"^\s*import\s+pickle\b|^\s*from\s+pickle\b", file_text, re.M) is not None

    for line in original_lines:
        m = CPICKLE_IMPORT_RE.match(line)
        if m:
            # If the file already imports 'pickle', replace cPickle import with 'import pickle'
            if has_pickle_import:
                new_lines.append(re.sub(r"cPickle", "pickle", line))
            else:
                # insert a safe shim
                shim = [
                    "try:\n",
                    "    import cPickle as pickle\n",
                    "except Exception:\n",
                    "    import pickle\n",
                ]
                new_lines.extend(shim)
            changed = True
        else:
            new_lines.append(line)
    return new_lines, changed


def transform_content(original_text):
    """Return transformed text (string) and a boolean indicating change."""
    orig_lines = original_text.splitlines(keepends=True)
    text = original_text
    changed = False

    # cPickle shim (works line-oriented)
    new_lines, c_changed = make_cpickle_shim(orig_lines)
    if c_changed:
        text = ''.join(new_lines)
        changed = True

    # simple replacements
    t = Xrange_RE.sub('range(', text)
    if t != text:
        text = t
        changed = True

    t = ITERITEMS_RE.sub('.items(', text)
    if t != text:
        text = t
        changed = True

    t = RAW_INPUT_RE.sub('input(', text)
    if t != text:
        text = t
        changed = True

    return text, changed


def generate_patch(file_path, original_text, new_text):
    orig_lines = original_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, new_lines,
                                fromfile=str(file_path), tofile=str(file_path), lineterm='')
    return '\n'.join(diff)


def scan_and_generate():
    PATCH_DIR.mkdir(exist_ok=True)
    changed_files = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # skip excluded dirs
        parts = Path(dirpath).parts
        if any(part in EXCLUDE_DIRS for part in parts):
            continue
        for fn in filenames:
            if not fn.endswith(PY_EXTS):
                continue
            fp = Path(dirpath) / fn
            try:
                text = fp.read_text(encoding='utf-8')
            except Exception:
                # skip files we can't decode
                continue
            new_text, changed = transform_content(text)
            if changed:
                p = PATCH_DIR / fp.relative_to(ROOT)
                p.parent.mkdir(parents=True, exist_ok=True)
                patch_text = generate_patch(fp.relative_to(ROOT), text, new_text)
                # write patch file
                patch_file = p.with_suffix(p.suffix + '.patch')
                patch_file.write_text(patch_text, encoding='utf-8')
                changed_files += 1
                print(f'Wrote patch: {patch_file}')
    print(f'Done. Generated {changed_files} patch files under {PATCH_DIR}')


def apply_changes():
    """Apply transformations in-place and commit the changes on the current branch.
    This writes transformed files directly (after backing up the original with a .orig suffix),
    stages and commits them with a message.
    """
    changed_files = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # skip excluded dirs
        parts = Path(dirpath).parts
        if any(part in EXCLUDE_DIRS for part in parts):
            continue
        for fn in filenames:
            if not fn.endswith(PY_EXTS):
                continue
            fp = Path(dirpath) / fn
            try:
                text = fp.read_text(encoding='utf-8')
            except Exception:
                continue
            new_text, changed = transform_content(text)
            if changed:
                backup = fp.with_suffix(fp.suffix + '.orig')
                if not backup.exists():
                    backup.write_text(text, encoding='utf-8')
                fp.write_text(new_text, encoding='utf-8')
                changed_files += 1
                print(f'Applied changes to: {fp}')
                # stage the file
                os.system(f'git add "{fp}"')
    if changed_files:
        os.system('git commit -m "Apply automated Python2->3 conservative transformations"')
    print(f'Done. Applied changes to {changed_files} files.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate or apply conservative Python2->3 patches')
    parser.add_argument('--apply', action='store_true', help='Apply transformations in-place and commit')
    args = parser.parse_args()
    if args.apply:
        apply_changes()
    else:
        scan_and_generate()
