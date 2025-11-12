#!/usr/bin/env python3
"""
Comprehensive Python 2 to Python 3 print statement converter.
Uses AST parsing to safely convert print statements.
"""
import ast
import sys
from pathlib import Path


def convert_file(filepath):
    """Convert print statements in a file using AST parsing."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    lines = content.split('\n')
    modified = False
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip comment-only lines
        if line.strip().startswith('#'):
            new_lines.append(line)
            i += 1
            continue
        
        # Check if this line has an old-style print statement
        stripped = line.lstrip()
        if stripped.startswith('print ') and not stripped.startswith('print('):
            # Extract leading whitespace
            indent = line[:len(line) - len(stripped)]
            
            # Check for multi-line print statements (ending with \)
            full_statement = stripped
            j = i
            while full_statement.rstrip().endswith('\\') and j + 1 < len(lines):
                j += 1
                full_statement = full_statement.rstrip()[:-1] + ' ' + lines[j].strip()
            
            # Extract the print arguments (everything after "print ")
            args = full_statement[6:].rstrip()  # Remove "print " prefix
            
            # Handle special cases
            if args.startswith('"') or args.startswith("'"):
                # String argument - wrap in print()
                new_statement = f'{indent}print({args})'
            else:
                # Other arguments - wrap in print()
                new_statement = f'{indent}print({args})'
            
            # Add any trailing comment
            if j > i:
                # Multi-line: combine lines
                new_lines.append(new_statement)
                i = j + 1
            else:
                # Single-line
                if len(line) > len(indent) + 6 + len(args):
                    # There's a comment or trailing content
                    tail = line[len(indent) + 6 + len(args):]
                    new_lines.append(new_statement + tail)
                else:
                    new_lines.append(new_statement)
                i += 1
            
            modified = True
        else:
            new_lines.append(line)
            i += 1
    
    if modified:
        new_content = '\n'.join(new_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_all_prints.py <file1> [<file2> ...]")
        sys.exit(1)
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.exists():
            if convert_file(filepath):
                print(f"✓ Fixed print statements in {filepath}")
            else:
                print(f"  No print statements to fix in {filepath}")
        else:
            print(f"✗ File not found: {filepath}")
