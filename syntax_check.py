#!/usr/bin/env python3
"""
Basic syntax check for app.py
"""

import ast
import sys

def check_syntax(filename):
    """Check Python syntax of a file"""
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        print(f"✅ {filename} - Syntax is valid!")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename} - Syntax error:")
        print(f"  Line {e.lineno}: {e.text}")
        print(f"  Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {filename} - Error reading file: {e}")
        return False

if __name__ == "__main__":
    success = check_syntax("app.py")
    sys.exit(0 if success else 1)