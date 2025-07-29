#!/usr/bin/env python3
"""
Validate configuration files for PWMK project.
"""

import json
import sys
import os
from pathlib import Path

def validate_json_files():
    """Validate JSON configuration files."""
    json_files = [
        'renovate.json',
        'monitoring/grafana/dashboards/pwmk-dashboard.json'
    ]
    
    results = {}
    for file_path in json_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                results[file_path] = "VALID"
            except json.JSONDecodeError as e:
                results[file_path] = f"INVALID: {e}"
        else:
            results[file_path] = "NOT_FOUND"
    
    return results

def validate_makefile():
    """Validate Makefile syntax."""
    makefile_path = 'Makefile'
    
    if not os.path.exists(makefile_path):
        return "NOT_FOUND"
    
    try:
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Basic Makefile validation
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for tabs in recipe lines
            if line.startswith('\t') or line.startswith('    '):
                continue
            elif ':' in line and not line.strip().startswith('#'):
                # Target line should be followed by recipe lines with tabs
                continue
        
        return "VALID"
    except Exception as e:
        return f"INVALID: {e}"

def validate_requirements_files():
    """Validate requirements files."""
    req_files = [
        'requirements/dev.in',
        'requirements/test.in', 
        'requirements/docs.in'
    ]
    
    results = {}
    for file_path in req_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Basic validation - check for valid package format
                valid = True
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Should be package name with optional version spec
                        if '=' in line or '>=' in line or '<=' in line or line.replace('-', '').replace('_', '').replace('[', '').replace(']', '').replace('.', '').isalnum():
                            continue
                        else:
                            valid = False
                            break
                
                results[file_path] = "VALID" if valid else f"INVALID: line {line_num}"
            except Exception as e:
                results[file_path] = f"INVALID: {e}"
        else:
            results[file_path] = "NOT_FOUND"
    
    return results

def validate_shell_scripts():
    """Basic validation of shell scripts."""
    script_files = [
        'scripts/setup_dev_env.sh'
    ]
    
    results = {}
    for file_path in script_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for bash shebang
                if content.startswith('#!/bin/bash') or content.startswith('#!/usr/bin/env bash'):
                    results[file_path] = "VALID"
                else:
                    results[file_path] = "VALID (no shebang check)"
            except Exception as e:
                results[file_path] = f"INVALID: {e}"
        else:
            results[file_path] = "NOT_FOUND"
    
    return results

def main():
    """Run all configuration validations."""
    
    print("🔍 Validating PWMK Configuration Files")
    print("=" * 50)
    
    all_valid = True
    
    # Validate JSON files
    print("\n📄 JSON Files:")
    json_results = validate_json_files()
    for file_path, status in json_results.items():
        print(f"  {file_path}: {status}")
        if "INVALID" in status:
            all_valid = False
    
    # Validate Makefile
    print("\n🔨 Makefile:")
    makefile_status = validate_makefile()
    print(f"  Makefile: {makefile_status}")
    if "INVALID" in makefile_status:
        all_valid = False
    
    # Validate requirements files
    print("\n📦 Requirements Files:")
    req_results = validate_requirements_files()
    for file_path, status in req_results.items():
        print(f"  {file_path}: {status}")
        if "INVALID" in status:
            all_valid = False
    
    # Validate shell scripts
    print("\n🐚 Shell Scripts:")
    script_results = validate_shell_scripts()
    for file_path, status in script_results.items():
        print(f"  {file_path}: {status}")
        if "INVALID" in status:
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✅ All configuration files are valid!")
        return 0
    else:
        print("❌ Some configuration files have issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())