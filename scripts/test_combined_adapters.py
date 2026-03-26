#!/usr/bin/env python3
"""
Test script for the new pruning hook functions with combined adapters.

This script tests the hook_pruning_process_info and get_pruning_process_info functions
with both LoRA and Bottleneck adapters to ensure they work correctly.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Import our modified functions
from peft.tuners.prunepeft.model import hook_pruning_process_info, get_pruning_process_info, compute_pruning_rankings
from peft import get_peft_model
from peft.tuners.prunepeft import PrunePEFTConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_model_and_data():
    """Create a simple test model and dataset."""
    # For testing purposes, we'll use a small model
    # In practice, you would load your actual model
    model_id = "/home/autopeft/LoRA-GA/ckpts/pretrained/Llama-2-7b-hf"

    try:
        # Try to load the actual model if available
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        logger.info("Loaded actual model")
    except:
        # Fallback to a simple model for testing
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        logger.info("Loaded fallback GPT2 model")

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a simple dataset
    input_ids = torch.randint(0, 1000, (10, 50))  # 10 samples, 50 tokens each
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    return model, tokenizer, dataloader

def create_combined_test_config(model):
    """Create a PrunePEFT configuration with both LoRA and Bottleneck adapters for testing."""
    config = PrunePEFTConfig(
        task_type="CAUSAL_LM",
        adapter_types=["lora", "bottleneck"],  # Combined mode
        r=8,
        lora_alpha=32,
        bottleneck_size=64,
        bottleneck_dropout=0.1,
        target_modules=None,  # Let the model decide target_modules for combined mode
        bias="none",
    )
    return config

def validate_data_quality(process_info):
    """Validate that the collected data is of good quality (not NaN, inf, etc.)."""
    logger.info("Validating data quality...")

    # Check gradients
    gradients = process_info.get("gradients", {})
    if not gradients:
        logger.warning("No gradients collected")
    else:
        logger.info(f"Checking {len(gradients)} gradient tensors...")
        lora_params = 0
        bottleneck_params = 0

        for name, grad in gradients.items():
            # Convert to numpy if it's not already
            if isinstance(grad, torch.Tensor):
                grad_np = grad.cpu().numpy()
            else:
                grad_np = grad

            # Count parameter types
            if "lora" in name:
                lora_params += 1
            if "bottleneck" in name:
                bottleneck_params += 1

            # Check for NaN values
            has_nan = np.isnan(grad_np).any()
            # Check for infinite values
            has_inf = np.isinf(grad_np).any()
            # Check for valid range
            grad_min, grad_max = grad_np.min(), grad_np.max()

            logger.info(f"  {name}: shape={grad_np.shape}, range=[{grad_min:.6f}, {grad_max:.6f}], "
                       f"has_nan={has_nan}, has_inf={has_inf}")

            if has_nan or has_inf:
                logger.error(f"Invalid gradient data in {name}")
                return False

        logger.info(f"  Total parameters: {len(gradients)}, LoRA: {lora_params}, Bottleneck: {bottleneck_params}")

    # Check activations
    activations = process_info.get("activations", {})
    if not activations:
        logger.warning("No activations collected")
    else:
        logger.info(f"Checking {len(activations)} activation tensors...")
        lora_acts = 0
        bottleneck_acts = 0

        for name, act in activations.items():
            # Convert to numpy if it's not already
            if isinstance(act, torch.Tensor):
                act_np = act.cpu().numpy()
            else:
                act_np = act

            # Count activation types
            if "lora" in name:
                lora_acts += 1
            if "bottleneck" in name:
                bottleneck_acts += 1

            # Check for NaN values
            has_nan = np.isnan(act_np).any()
            # Check for infinite values
            has_inf = np.isinf(act_np).any()
            # Check for valid range
            act_min, act_max = act_np.min(), act_np.max()

            logger.info(f"  {name}: shape={act_np.shape}, range=[{act_min:.6f}, {act_max:.6f}], "
                       f"has_nan={has_nan}, has_inf={has_inf}")

            if has_nan or has_inf:
                logger.error(f"Invalid activation data in {name}")
                return False

        logger.info(f"  Total activations: {len(activations)}, LoRA: {lora_acts}, Bottleneck: {bottleneck_acts}")

    logger.info("✓ Data quality validation passed")
    return True

def test_combined_pruning_hooks():
    """Test the new pruning hook functions with combined adapters."""
    logger.info("Starting combined adapters pruning hooks test...")

    # Create model and data
    model, tokenizer, dataloader = create_test_model_and_data()

    # Create PEFT config and wrap model with combined adapters
    peft_config = create_combined_test_config(model)
    model = get_peft_model(model, peft_config)

    logger.info("Model wrapped with PrunePEFT (combined LoRA and Bottleneck)")

    # Test 1: hook_pruning_process_info
    logger.info("Test 1: Setting up pruning hooks...")
    hook_pruning_process_info(
        model=model,
        adapter_name="default",
        opts=("lora", "adapter", "bottleneck")
    )
    logger.info("✓ hook_pruning_process_info executed successfully")

    # Test 2: Run a few training steps
    logger.info("Test 2: Running training steps...")
    model.train()

    # Run a single batch through the model
    for batch in dataloader:
        if len(batch) == 3:
            input_ids, attention_mask, labels = batch
        else:
            input_ids, attention_mask = batch
            labels = input_ids.clone()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Backward pass
        loss = outputs.loss
        loss.backward()

        # We only need one batch for testing
        break

    logger.info("✓ Training steps executed successfully")

    # Test 3: get_pruning_process_info
    logger.info("Test 3: Retrieving pruning process info...")
    process_info = get_pruning_process_info(model)

    # Verify the structure of process_info
    assert "gradients" in process_info, "process_info should contain 'gradients'"
    assert "activations" in process_info, "process_info should contain 'activations'"
    assert "trainable_param_names" in process_info, "process_info should contain 'trainable_param_names'"

    logger.info(f"✓ get_pruning_process_info executed successfully")
    logger.info(f"  - Number of gradients: {len(process_info['gradients'])}")
    logger.info(f"  - Number of activations: {len(process_info['activations'])}")
    logger.info(f"  - Number of trainable parameters: {len(process_info['trainable_param_names'])}")

    # Test 4: Validate data quality
    logger.info("Test 4: Validating data quality...")
    is_valid = validate_data_quality(process_info)
    assert is_valid, "Data quality validation failed"
    logger.info("✓ Data quality validation passed")

    # Test 5: compute_pruning_rankings
    logger.info("Test 5: Computing pruning rankings...")
    rankings = compute_pruning_rankings(
        model=model,
        adapter_name="default",
        opts=("lora", "adapter", "bottleneck"),
        process_info=process_info,
        top_p=2,
    )

    # Verify rankings structure
    expected_methods = ['zeros', 'values_below_threshold', 'gradient', 'activation', 'snip', 'minimum_weight']
    for method in expected_methods:
        assert method in rankings, f"rankings should contain '{method}'"
        logger.info(f"  - {method}: {len(rankings[method])} groups")

        # Check that scores are valid numbers
        for i, (prefix, names, score) in enumerate(rankings[method]):
            assert not np.isnan(score), f"Score for {method}[{i}] is NaN"
            assert not np.isinf(score), f"Score for {method}[{i}] is infinite"
            logger.info(f"    {i+1}. {prefix}: score={score:.6f}")

    logger.info("✓ compute_pruning_rankings executed successfully")

    logger.info("All tests passed! The pruning hook functions work correctly with combined adapters and produce valid data.")
    return True

if __name__ == "__main__":
    try:
        test_combined_pruning_hooks()
        print("\n✓ All tests passed!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise