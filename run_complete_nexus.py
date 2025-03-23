#!/usr/bin/env python
"""
Complete integrated runner for the NEXUS medical application
This script provides a unified interface for running the NEXUS system
on various medical datasets with full integration of all components.
"""

import os
import sys
import argparse
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

# Import our integration module
from nexus_medical_integration import (
    apply_medical_patches,
    load_medical_dataset,
    create_data_loaders,
    create_medical_knowledge_graph,
    run_scaled_nexus_experiment
)

def print_banner(text):
    """Print a nicely formatted banner"""
    width = len(text) + 10
    print("=" * width)
    print(f"    {text}")
    print("=" * width)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run NEXUS medical system with complete integration")
    
    # Add command-line arguments
    parser.add_argument("--dataset", type=str, choices=['heart_disease', 'diabetes', 'breast_cancer'], 
                        default='heart_disease', help="Medical dataset to use")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="nexus_results", 
                        help="Output directory for results")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate and save visualization plots")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with additional output")
    
    args = parser.parse_args()
    
    # Print welcome banner
    print_banner(f"NEXUS Medical System - {args.dataset.replace('_', ' ').title()}")
    
    # Apply integration patches to ensure all components work together
    print("\nApplying integration patches...")
    apply_medical_patches()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the experiment
    start_time = time.time()
    
    try:
        print("Attempting to run the experiment...")
        result = run_scaled_nexus_experiment(
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        if result is None:
            print("Error: Experiment returned None. Check the integration module.")
            return 1
            
        # Access the model and results
        if args.debug:
            print(f"Result type: {type(result)}")
            print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'None'}")
        
        model = result.get('model')
        if model is None:
            print("Error: Model not found in result dictionary.")
            print(f"Available keys: {list(result.keys()) if isinstance(result, dict) else 'None'}")
            return 1
        
        results = result.get('results')
        if results is None:
            print("Error: Results not found in result dictionary.")
            return 1
        
        # Export results to CSV
        try:
            csv_path = os.path.join(args.output_dir, f"{args.dataset}_results.csv")
            model.export_results(csv_path)
            print(f"\nResults exported to {csv_path}")
        except Exception as e:
            print(f"Warning: Could not export results to CSV: {e}")
            if args.debug:
                traceback.print_exc()
        
        # Generate visualizations if requested
        if args.visualize:
            try:
                print("\nGenerating visualizations...")
                vis_prefix = os.path.join(args.output_dir, f"{args.dataset}_vis")
                model.visualize_results(
                    output_prefix=vis_prefix,
                    save_figures=True,
                    show_figures=False
                )
                print(f"Visualizations saved to {vis_prefix}_*.png")
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")
                if args.debug:
                    traceback.print_exc()
        
        # Try to save the model
        try:
            model_path = os.path.join(args.output_dir, f"{args.dataset}_model")
            model.save_model(model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
            if args.debug:
                traceback.print_exc()
            
        # Try to run an example diagnosis
        try:
            print("\nRunning example diagnosis...")
            # Create an appropriate test sample based on the dataset
            if args.dataset == 'heart_disease':
                # Sample for heart disease with realistic values (age, sex, cp, trestbps, etc.)
                sample_input = torch.tensor([[60.0, 1.0, 3.0, 140.0, 260.0, 0.0, 1.0, 140.0, 0.0, 0.0, 1.0, 0.0, 3.0]], 
                                           dtype=torch.float32)
            elif args.dataset == 'diabetes':
                # Sample for diabetes (pregnancies, glucose, bp, skin, insulin, bmi, dpf, age)
                sample_input = torch.tensor([[0.0, 150.0, 80.0, 30.0, 100.0, 28.5, 0.5, 50.0]], 
                                           dtype=torch.float32)
            elif args.dataset == 'breast_cancer':
                # For breast cancer, use random values as it has many features
                feature_count = len(model.symbol_names)
                sample_input = torch.rand((1, feature_count), dtype=torch.float32)
            else:
                # Fallback to random values
                feature_count = len(model.symbol_names)
                sample_input = torch.rand((1, feature_count), dtype=torch.float32)
            
            # Run diagnosis
            diagnosis = model.diagnose(sample_input, risk_level='medium')
            
            # Print diagnosis results
            print("\nExample Diagnosis Results:")
            print(f"Predicted Class: {diagnosis['nexus']['class_name']}")
            print(f"Confidence: {diagnosis['nexus']['confidence']:.2f}")
            print(f"Strategy Used: {diagnosis['nexus']['strategy']['strategy']}")
            
            # Print explanation
            print("\nExplanation:")
            explanation = model.explain_diagnosis(diagnosis, detail_level='medium')
            print(explanation)
        except Exception as e:
            print(f"Warning: Could not run example diagnosis: {e}")
            if args.debug:
                traceback.print_exc()
            
    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}")
        if args.debug:
            traceback.print_exc()
        else:
            print("Run with --debug for more detailed error information")
        return 1
        
    # Print completion message
    total_time = time.time() - start_time
    print_banner(f"NEXUS experiment completed in {total_time/60:.2f} minutes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())