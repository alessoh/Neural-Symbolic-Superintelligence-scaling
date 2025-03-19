# NEXUS Scaling Migration Guide


## Overview of Changes

The scaled NEXUS implementation introduces several significant enhancements:

1. **Enhanced Core Model Efficiency**
   - Optimized transformer layers with memory-efficient attention mechanisms
   - Gradient checkpointing for larger batch training
   - Mixed precision training (FP16/BF16) support

2. **Knowledge Graph Scaling**
   - Sparse representation for large knowledge graphs
   - Distributed storage for graphs with millions of nodes
   - Hierarchical indexing for faster entity and relation lookups

3. **Parallel Computation Support**
   - Multi-GPU support using PyTorch DistributedDataParallel
   - Model parallelism for extremely large models
   - Asynchronous symbolic reasoning that runs parallel to neural computation

## Migration Steps

### Step 1: Install Required Dependencies

```bash
pip install torch>=1.10.0 numpy>=1.20.0 scipy>=1.7.0 networkx>=2.6.0 tqdm>=4.62.0
```

For multi-GPU support, ensure you have PyTorch built with CUDA support:

```bash
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 2: Import the New Modules

Replace the original imports with the scaled versions:

```python
# Original imports
from nexus_real_data import EnhancedNEXUSModel, AdvancedNeuralModel, EnhancedKnowledgeGraph

# New imports
from efficient_transformer import ScalableTransformerModel, MixedPrecisionTrainer
from sparse_knowledge_graph import SparseKnowledgeGraph, DistributedKnowledgeGraph
from parallel_computation import (
    NEXUSDistributedTrainer, ModelParallelNEXUS, AsyncSymbolicReasoner,
    NEXUSParallelExecutor, NEXUSParallelExecutionPlan, NEXUSPerformanceProfiler
)
from nexus_scaling_integration import ScalableNEXUSModel
```

### Step 3: Migrate Your Dataset

If you're using a custom dataset, you'll need to modify your data loading code to leverage the parallel data loading capabilities:

```python
# Original data loading
train_dataset = MedicalDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Scaled data loading (for distributed training)
def create_data_loaders(dataset, batch_size, distributed=False, world_size=1, rank=0):
    if distributed:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True
        )
    else:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
```

### Step 4: Migrate the NEXUS Model

Replace your existing NEXUS model instantiation with the scaled version:

```python
# Original model
model = EnhancedNEXUSModel(
    input_dim=input_dim,
    num_classes=num_classes,
    num_symbols=num_symbols,
    symbol_names=symbol_names,
    class_names=class_names,
    embed_dim=128
)

# Scaled model
model = ScalableNEXUSModel(
    input_dim=input_dim,
    num_classes=num_classes,
    num_symbols=num_symbols,
    symbol_names=symbol_names,
    class_names=class_names,
    embed_dim=128,
    device="cuda",  # Use GPU if available
    use_mixed_precision=True,  # Enable mixed precision for faster training
    use_sparse_knowledge_graph=True,  # Enable sparse knowledge graph
    async_symbolic_reasoning=True,  # Enable asynchronous symbolic reasoning
    num_symbolic_workers=2,  # Number of workers for symbolic reasoning
    model_parallel=False,  # Set to True for multi-GPU model parallelism
    num_gpus=1,  # Number of GPUs for model parallelism
    profiling=True  # Enable performance profiling
)
```

### Step 5: Migrate the Training Code

Update your training code to use the enhanced training capabilities:

```python
# Original training
model.train_neural(train_loader, num_epochs=10, lr=0.001)

# Scaled training
# Option 1: Use the built-in trainer
model.train_neural(
    train_loader,
    num_epochs=10,
    lr=0.001,
    scheduler='cosine',  # Use cosine learning rate scheduler
    weight_decay=1e-5  # Add weight decay for regularization
)

# Option 2: For distributed training across multiple GPUs
trainer = NEXUSDistributedTrainer(
    model,
    world_size=torch.cuda.device_count(),
    backend="nccl",
    checkpoint_dir="checkpoints"
)
model = trainer.prepare_model()
train_loader = trainer.prepare_dataloader(train_dataset, batch_size=32)
model.train_neural(train_loader, num_epochs=10, lr=0.001)
```

### Step 6: Migrate the Inference/Diagnosis Code

Update your inference code to leverage the asynchronous capabilities:

```python
# Original diagnosis
result = model.diagnose(input_data, active_symptoms)

# Scaled diagnosis (synchronous)
result = model.diagnose(input_data, active_symptoms)

# Scaled diagnosis (asynchronous) - returns immediately with neural results
# and continues symbolic reasoning in the background
result = model.diagnose_async(input_data, active_symptoms)

# Check if an asynchronous result is ready
request_id = result['symbolic']['request_id']
complete_result = model.check_pending_diagnosis(request_id)

# Wait for all pending diagnoses to complete
all_results = model.wait_for_pending_diagnoses(timeout=10.0)  # Timeout in seconds
```

### Step 7: Migrate the Evaluation Code

Update your evaluation code to use the enhanced parallel evaluation:

```python
# Original evaluation
test_results = model.evaluate(
    test_loader,
    symptom_dict=symptom_dict,
    feedback=True
)

# Scaled evaluation
test_results = model.evaluate(
    test_loader,
    symptom_dict=symptom_dict,
    feedback=True,
    use_async=True,  # Enable asynchronous evaluation
    num_workers=4  # Number of parallel workers for evaluation
)
```

### Step 8: Leverage the Advanced Execution Plan for Batch Processing

For processing large datasets, use the parallel execution plan:

```python
from parallel_computation import NEXUSParallelExecutionPlan

executor = NEXUSParallelExecutor(
    neural_model=model.neural_model,
    knowledge_graph=model.knowledge_graph,
    device=model.device,
    num_symbolic_workers=4,
    enable_async=True
)

execution_plan = NEXUSParallelExecutionPlan(
    executor=executor,
    batch_size=32,
    max_pending=100,
    max_workers=4
)

# Process a batch of inputs
results = execution_plan.process_batch(
    inputs=batch_inputs,
    active_symptoms_list=batch_symptoms,
    risk_levels=batch_risk_levels
)

# Process an entire dataset
results = execution_plan.process_dataset(
    dataloader=test_loader,
    active_symptoms_dict=symptom_dict,
    risk_level_dict=risk_dict
)
```

### Step 9: Profile Performance to Identify Bottlenecks

Use the performance profiler to identify bottlenecks:

```python
profiler = NEXUSPerformanceProfiler()

# Start timing an operation
profiler.start_timer("my_operation")

# Do some work
# ...

# Stop timing and record duration
duration = profiler.stop_timer("my_operation")

# Increment a counter
profiler.increment_counter("processed_items")

# Get timing statistics
avg_time = profiler.get_average_time("my_operation")
total_time = profiler.get_total_time("my_operation")

# Print a performance summary
profiler.print_summary()
```

### Step 10: Save and Load the Scaled Model

The scaled model provides enhanced saving and loading capabilities:

```python
# Save the model
model.save_model("saved_model_directory")

# Load the model
loaded_model = ScalableNEXUSModel.load_model(
    "saved_model_directory",
    device="cuda"  # Load on GPU
)
```

## Feature-by-Feature Migration Guide

### Mixed Precision Training

To enable mixed precision training for faster computation and reduced memory usage:

```python
# Enable mixed precision when creating the model
model = ScalableNEXUSModel(
    # ... other parameters ...
    use_mixed_precision=True
)

# Or use the MixedPrecisionTrainer directly
from efficient_transformer import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(
    model=model.neural_model,
    optimizer=optimizer,
    use_mixed_precision=True,
    use_checkpointing=True  # Enable gradient checkpointing
)

# Training step with mixed precision
loss = trainer.training_step(inputs, targets, criterion)
```

### Sparse Knowledge Graph

To migrate to a sparse knowledge graph for efficient memory usage:

```python
# Initialize sparse knowledge graph when creating the model
model = ScalableNEXUSModel(
    # ... other parameters ...
    use_sparse_knowledge_graph=True,
    knowledge_graph_storage_dir="knowledge_graph_storage"  # Optional storage directory
)

# Or create the sparse knowledge graph directly
from sparse_knowledge_graph import SparseKnowledgeGraph

kg = SparseKnowledgeGraph(
    name="my_knowledge_graph",
    storage_dir="knowledge_graph_storage"  # Optional
)

# Adding entities, relations and rules works the same way
kg.add_entity(entity_id=1, name="entity_name", attributes={"type": "symptom"})
kg.add_relation(source_id=1, relation_type="indicates", target_id=2, weight=0.9)
kg.add_rule([1, 2], 3, confidence=0.8)
kg.add_hierarchy(child_id=1, parent_id=2)

# New functionality for large graphs
# Create a subgraph of only relevant entities
subgraph = kg.create_subgraph([1, 2, 3, 4])

# Convert to NetworkX for visualization
import networkx as nx
nx_graph = kg.convert_to_networkx()

# Save and load from disk
kg.save_to_disk("knowledge_graph_storage")
loaded_kg = SparseKnowledgeGraph.load_from_disk("knowledge_graph_storage")
```

### Distributed Knowledge Graph

For extremely large knowledge graphs that need to be distributed across multiple storage nodes:

```python
from sparse_knowledge_graph import DistributedKnowledgeGraph

# Initialize a distributed knowledge graph with 4 partitions
dkg = DistributedKnowledgeGraph(
    name="distributed_graph",
    storage_dir="distributed_storage",
    num_partitions=4
)

# Usage is similar to the regular knowledge graph
dkg.add_entity(entity_id=1, name="entity_name", attributes={"type": "symptom"})
dkg.add_relation(source_id=1, relation_type="indicates", target_id=2, weight=0.9)
dkg.add_rule([1, 2], 3, confidence=0.8)
dkg.add_hierarchy(child_id=1, parent_id=2)

# Reasoning works the same way
inferred, reasoning_steps, confidences, class_scores = dkg.reason(active_entities, max_hops=3)

# Save and load distributed graph
dkg.save_to_disk()
loaded_dkg = DistributedKnowledgeGraph.load_from_disk("distributed_storage")
```

### Asynchronous Symbolic Reasoning

To enable asynchronous symbolic reasoning that runs in parallel to neural computation:

```python
# Enable when creating the model
model = ScalableNEXUSModel(
    # ... other parameters ...
    async_symbolic_reasoning=True,
    num_symbolic_workers=4  # Number of worker threads
)

# Or create the async reasoner directly
from parallel_computation import AsyncSymbolicReasoner

reasoner = AsyncSymbolicReasoner(
    knowledge_graph=knowledge_graph,
    num_workers=4
)

# Start the worker threads
reasoner.start()

# Submit a reasoning task
request_id = reasoner.submit_reasoning_task(
    request_id=1,
    active_entities=[1, 2, 3],
    max_hops=3
)

# Get the result when ready
result = reasoner.get_result_for_request(request_id, timeout=5.0)

# Wait for all pending requests to complete
all_results = reasoner.get_result()

# Stop the workers when done
reasoner.stop()
```

### Multi-GPU Training

To distribute training across multiple GPUs:

```python
from parallel_computation import NEXUSDistributedTrainer
import torch.distributed as dist
import torch.multiprocessing as mp

def train_distributed(rank, world_size, args):
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    model = ScalableNEXUSModel(
        # ... other parameters ...
        device=f"cuda:{rank}"
    )
    
    # Create distributed trainer
    trainer = NEXUSDistributedTrainer(
        model=model,
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        checkpoint_dir="checkpoints"
    )
    
    # Prepare model and data
    ddp_model = trainer.prepare_model()
    train_loader = trainer.prepare_dataloader(train_dataset, batch_size=32)
    
    # Train the model
    for epoch in range(args.epochs):
        ddp_model.train_neural(train_loader, num_epochs=1, lr=args.lr)
        
        # Save checkpoints
        trainer.save_checkpoint(
            epoch=epoch,
            optimizer=optimizer,
            metrics={"accuracy": accuracy}
        )
    
    # Clean up
    dist.destroy_process_group()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_distributed,
        args=(world_size, args),
        nprocs=world_size
    )
```

### Model Parallelism

For extremely large models that don't fit on a single GPU:

```python
from parallel_computation import ModelParallelNEXUS, create_optimal_device_map

# Create a device map that distributes the model across GPUs
num_gpus = torch.cuda.device_count()
device_map = create_optimal_device_map(model.neural_model, num_gpus)

# Alternative: Manual device map
device_map = {
    'embedding': 0,
    'transformer_layers.0': 0,
    'transformer_layers.1': 1,
    'transformer_layers.2': 2,
    'classifier': 3
}

# Create model parallel wrapper
model_parallel = ModelParallelNEXUS(
    neural_model=model.neural_model,
    device_map=device_map,
    pipeline_chunks=2  # Number of chunks for pipeline parallelism
)

# Forward pass with model parallelism
outputs = model_parallel.forward(inputs, use_pipeline=True)
```

## Benchmarking and Performance Tuning

After migration, it's essential to benchmark your implementation to ensure you're getting the expected performance improvements. Here's a simple benchmarking approach:

```python
import time

def benchmark_model(model, test_loader, num_runs=10):
    # Warmup
    model.evaluate(test_loader)
    
    # Benchmark
    neural_times = []
    symbolic_times = []
    total_times = []
    
    for _ in range(num_runs):
        # Neural component timing
        start = time.time()
        for inputs, _ in test_loader:
            with torch.no_grad():
                model.neural_model(inputs.to(model.device))
        neural_times.append(time.time() - start)
        
        # Full model timing
        start = time.time()
        model.evaluate(test_loader)
        total_times.append(time.time() - start)
        
        # Estimate symbolic time
        symbolic_times.append(total_times[-1] - neural_times[-1])
    
    # Calculate averages
    avg_neural = sum(neural_times) / num_runs
    avg_symbolic = sum(symbolic_times) / num_runs
    avg_total = sum(total_times) / num_runs
    
    print(f"Neural component: {avg_neural:.4f}s")
    print(f"Symbolic component: {avg_symbolic:.4f}s")
    print(f"Total execution: {avg_total:.4f}s")
    
    return {
        "neural": avg_neural,
        "symbolic": avg_symbolic,
        "total": avg_total
    }

# Compare original and scaled implementations
original_times = benchmark_model(original_model, test_loader)
scaled_times = benchmark_model(scaled_model, test_loader)

# Calculate speedup
neural_speedup = original_times["neural"] / scaled_times["neural"]
symbolic_speedup = original_times["symbolic"] / scaled_times["symbolic"]
total_speedup = original_times["total"] / scaled_times["total"]

print(f"Neural speedup: {neural_speedup:.2f}x")
print(f"Symbolic speedup: {symbolic_speedup:.2f}x")
print(f"Total speedup: {total_speedup:.2f}x")
```

## Potential Challenges and Solutions

### Memory Issues with Large Knowledge Graphs

**Challenge**: Out-of-memory errors when loading large knowledge graphs.

**Solution**: Use the distributed knowledge graph with disk offloading:

```python
kg = DistributedKnowledgeGraph(
    name="large_graph",
    storage_dir="graph_storage",
    num_partitions=8  # Increase for larger graphs
)
kg.use_disk_offloading = True  # Enable disk offloading for very large graphs
```

### Slow Symbolic Reasoning

**Challenge**: Symbolic reasoning becomes a bottleneck for real-time applications.

**Solution**: Use asynchronous reasoning and increase the number of workers:

```python
model = ScalableNEXUSModel(
    # ... other parameters ...
    async_symbolic_reasoning=True,
    num_symbolic_workers=8  # Increase for faster reasoning
)

# Use asynchronous diagnosis that returns immediately with neural results
result = model.diagnose_async(input_data, active_symptoms)
```

### GPU Out of Memory with Large Batches

**Challenge**: GPU runs out of memory with large batch sizes.

**Solution**: Enable gradient checkpointing and mixed precision training:

```python
model = ScalableNEXUSModel(
    # ... other parameters ...
    use_mixed_precision=True  # Use FP16 instead of FP32
)

# Train with gradient checkpointing
model.train_neural(
    train_loader,
    use_checkpoint=True  # Enable gradient checkpointing
)
```

### Distributed Training Communication Overhead

**Challenge**: Slow training due to communication overhead in distributed training.

**Solution**: Optimize batch size and use gradient accumulation:

```python
# Increase effective batch size with gradient accumulation
accumulation_steps = 4
for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Conclusion

