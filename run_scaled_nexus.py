import torch
import random
import numpy as np
from nexus_scaling_integration import run_scaled_nexus_experiment

# Run the experiment with scikit-learn/iris dataset
# The updated nexus_scaling_integration.py now has special handling for the Iris dataset
result = run_scaled_nexus_experiment(
    dataset_name="scikit-learn/iris",  # This will now be handled correctly
    max_samples=1000,
    num_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    output_dir="nexus_results",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_mixed_precision=True,
    use_sparse_kg=True,
    async_reasoning=True,
    random_state=42
)

# Print summary
print(f"Training time: {result['train_time'] / 60:.2f} minutes")
print(f"NEXUS accuracy: {result['test_results']['nexus']['accuracy'] * 100:.2f}%")