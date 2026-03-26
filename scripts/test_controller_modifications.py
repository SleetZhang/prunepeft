#!/usr/bin/env python3
"""
Simple test script to verify the controller modifications work correctly.
"""

import sys
import os

# Add the project root to the path so we can import the controller
sys.path.insert(0, '/home/autopeft/LoRA-GA')

def test_imports():
    """Test that all required imports work correctly."""
    try:
        # Test importing the modified functions
        from peft.tuners.prunepeft.model import hook_pruning_process_info, get_pruning_process_info, compute_pruning_rankings
        print("✓ Successfully imported new pruning functions")

        # Test importing controller
        import examples.controller
        print("✓ Successfully imported controller module")

        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def check_function_replacements():
    """Check that the function replacements are correct."""
    try:
        # Read the controller file
        with open('/home/autopeft/LoRA-GA/examples/controller.py', 'r') as f:
            content = f.read()

        # Check that collect_pruning_process_info is no longer imported
        if 'collect_pruning_process_info' in content and 'from peft.tuners.prunepeft.model import' in content:
            # But allow it in comments
            lines = content.split('\n')
            import_lines = [line for line in lines if 'collect_pruning_process_info' in line and 'from peft.tuners.prunepeft.model import' in line]
            if import_lines:
                print(f"✗ Still importing collect_pruning_process_info: {import_lines[0]}")
                return False

        # Check that new functions are imported
        if 'hook_pruning_process_info, get_pruning_process_info, compute_pruning_rankings' not in content:
            print("✗ New functions not properly imported")
            return False

        # Check that the new functions are used
        if 'hook_pruning_process_info(' not in content:
            print("✗ hook_pruning_process_info not used in controller")
            return False

        if 'get_pruning_process_info(' not in content:
            print("✗ get_pruning_process_info not used in controller")
            return False

        print("✓ Function replacements appear correct")
        return True
    except Exception as e:
        print(f"✗ Function replacement check failed: {e}")
        return False

def main():
    """Main test function."""
    print("Testing controller modifications...")

    tests = [
        test_imports,
        check_function_replacements,
    ]

    all_passed = True
    for test in tests:
        if not test():
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed! Controller modifications are correct.")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())