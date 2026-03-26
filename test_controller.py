#!/usr/bin/env python3
"""
Test script to verify the modified controller.py works correctly
with direct torch model inference instead of vLLM.
"""

import sys
import os
import torch
from examples.controller import main

if __name__ == "__main__":
    # Test the modification with a small example
    # We'll use stage 3 (eval) only to test the inference logic

    print("Testing controller with direct torch model inference...")

    # Test case: Just run evaluation on a small subset
    # This should use the new torch inference, not vLLM
    test_args = [
        "--stage", "3",  # Only evaluation
        "--test_dataset", "gsm8k",
        "--sample_size", "8",  # Small sample for quick test
        "--model_path", ""  # Empty to force torch model inference
    ]

    # Simulate the argument parsing
    sys.argv = ["controller.py"] + test_args

    try:
        main()
        print("✅ Controller test completed successfully")
    except Exception as e:
        print(f"❌ Controller test failed: {e}")
        import traceback
        traceback.print_exc()