#!/usr/bin/env python3
"""
Simple Python 2 to Python 3 print statement converter.
Handles most common patterns but may need manual review for complex cases.
"""
import re
import sys
from pathlib import Path

def fix_print_statements(filepath):
    """Convert Python 2 print statements to Python 3 print() function calls."""
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        # Skip comment-only lines
        if line.strip().startswith('#'):
            new_lines.append(line)
            continue
            
        # Skip lines that are already using print as a function
        if 'print(' in line:
            new_lines.append(line)
            continue
        
        # Find print statements that aren't function calls
        # Pattern: "    print something" (with leading whitespace)
        match = re.match(r'^(\s*)print\s+(.+?)(\s*)(?:#.*)?$', line)
        if match:
            indent = match.group(1)
            args = match.group(2).rstrip()
            comment_part = line[len(indent) + len('print ') + len(args):].lstrip()
            
            # Skip if it looks like print is being used as a function or variable
            if '(' in args or not args or args[0] in ['(', '.']:
                new_lines.append(line)
                continue
            
            # Convert to print() function call
            new_line = f'{indent}print({args})'
            if comment_part:
                new_line += comment_part
            else:
                new_line += '\n'
            
            new_lines.append(new_line)
            modified = True
        else:
            new_lines.append(line)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_prints.py <file1> [<file2> ...]")
        sys.exit(1)
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.exists():
            if fix_print_statements(filepath):
                print(f"✓ Fixed print statements in {filepath}")
            else:
                print(f"  No print statements to fix in {filepath}")
        else:
            print(f"✗ File not found: {filepath}")
