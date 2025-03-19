import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict  # Make sure this import is added
from tqdm import tqdm  # Add this for progress bars

# Import enhanced components
from efficient_transformer import (
    ScalableTransformerModel, 
    MixedPrecisionTrainer
)
from sparse_knowledge_graph import (
    SparseKnowledgeGraph,
    DistributedKnowledgeGraph
)
from parallel_computation import (
    NEXUSDistributedTrainer,
    ModelParallelNEXUS,
    AsyncSymbolicReasoner,
    NEXUSParallelExecutor,
    NEXUSParallelExecutionPlan,
    NEXUSPerformanceProfiler,
    create_optimal_device_map
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NEXUS-Scaling")

class ScalableNEXUSModel(nn.Module):
    """
    Enhanced NEXUS model with scaling capabilities for handling large datasets
    and models across multiple devices.
    """
    def __init__(
        self, 
        input_dim: int,
        num_classes: int,
        num_symbols: int,
        symbol_names: List[str],
        class_names: List[str],
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 512,
        device: str = "cpu",
        use_mixed_precision: bool = False,
        use_flash_attention: bool = False,
        use_sparse_knowledge_graph: bool = True,
        distributed_knowledge_graph: bool = False,
        num_knowledge_partitions: int = 4,
        knowledge_graph_storage_dir: Optional[str] = None,
        async_symbolic_reasoning: bool = True,
        num_symbolic_workers: int = 2,
        model_parallel: bool = False,
        num_gpus: int = 1,
        profiling: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_symbols = num_symbols
        self.symbol_names = symbol_names
        self.class_names = class_names
        self.symbol_to_id = {name: i for i, name in enumerate(symbol_names)}
        self.device_str = device
        self.embed_dim = embed_dim
        self.use_mixed_precision = use_mixed_precision
        self.async_symbolic_reasoning = async_symbolic_reasoning
        self.model_parallel = model_parallel
        self.num_gpus = num_gpus
        
        # Move to device if needed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize enhanced neural model
        self.neural_model = ScalableTransformerModel(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            use_flash_attention=use_flash_attention,
            use_mixed_precision=use_mixed_precision
        )
        
        # Initialize knowledge graph
        if distributed_knowledge_graph:
            self.knowledge_graph = DistributedKnowledgeGraph(
                name="nexus_distributed_kg",
                storage_dir=knowledge_graph_storage_dir,
                num_partitions=num_knowledge_partitions
            )
        elif use_sparse_knowledge_graph:
            self.knowledge_graph = SparseKnowledgeGraph(
                name="nexus_sparse_kg",
                storage_dir=knowledge_graph_storage_dir
            )
        else:
            # Import original knowledge graph for compatibility
            from nexus_real_data import EnhancedKnowledgeGraph
            self.knowledge_graph = EnhancedKnowledgeGraph()
            
        self.knowledge_graph.symbol_offset = num_symbols
        self.knowledge_graph.num_classes = num_classes
        
        # Initialize neural-symbolic interface
        # Import from original code for now
        from nexus_real_data import AdvancedNeuralSymbolicInterface
        self.interface = AdvancedNeuralSymbolicInterface(
            hidden_dim=embed_dim,
            num_symbols=num_symbols,
            num_classes=num_classes
        )
        
        # Initialize metacognitive controller
        # Import from original code for now
        from nexus_real_data import AdvancedMetacognitiveController
        self.metacognitive = AdvancedMetacognitiveController()
        
        # Set up model parallelism if requested
        if model_parallel and num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
            # Create device map
            device_map = create_optimal_device_map(self.neural_model, num_gpus)
            self.model_parallel_wrapper = ModelParallelNEXUS(
                self.neural_model,
                device_map=device_map,
                pipeline_chunks=2
            )
            logger.info(f"Model parallelism enabled across {num_gpus} GPUs")
        else:
            self.model_parallel_wrapper = None
            # Move model to device
            self.neural_model = self.neural_model.to(self.device)
            self.interface = self.interface.to(self.device)
            
        # Set up async symbolic reasoning if requested
        if async_symbolic_reasoning:
            self.symbolic_reasoner = AsyncSymbolicReasoner(
                self.knowledge_graph,
                num_workers=num_symbolic_workers
            )
            self.symbolic_reasoner.start()
            logger.info(f"Async symbolic reasoning enabled with {num_symbolic_workers} workers")
        else:
            self.symbolic_reasoner = None
            
        # Set up parallel executor
        self.parallel_executor = NEXUSParallelExecutor(
            neural_model=self.neural_model if self.model_parallel_wrapper is None else self.model_parallel_wrapper,
            knowledge_graph=self.knowledge_graph,
            device=self.device,
            num_symbolic_workers=num_symbolic_workers,
            enable_async=async_symbolic_reasoning
        )
        
        # Set up performance profiler if requested
        if profiling:
            self.profiler = NEXUSPerformanceProfiler()
        else:
            self.profiler = None
            
        # Initialize results tracking
        self.eval_results = {
            'neural': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'symbolic': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'nexus': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []}
        }
        
        self.case_details = []
        
    def init_knowledge_graph(self):
        """Initialize the knowledge graph with entities and relationships"""
        kg = self.knowledge_graph
        
        # Add symbols (entities)
        for i, name in enumerate(self.symbol_names):
            kg.add_entity(i, name)
            
        # Add classes with offset
        offset = self.num_symbols
        for i, name in enumerate(self.class_names):
            kg.add_entity(offset + i, name)
            
        # Add example relations (can be extended)
        if self.num_symbols > 0 and self.num_classes > 0:
            kg.add_relation(0, "indicates", offset + 0, weight=0.9)
            if self.num_symbols > 1:
                kg.add_rule([0, 1], offset + 1, confidence=0.85)
                
        return kg
    
    def forward(self, x):
        """Forward pass using the appropriate neural model"""
        # Use model parallel wrapper if available
        if self.model_parallel_wrapper is not None:
            return self.model_parallel_wrapper.forward(x)
        else:
            x = x.to(self.device)
            return self.neural_model(x)
    
    def train_neural(self, dataloader, num_epochs=5, lr=0.001, scheduler=None, weight_decay=1e-5):
        """Enhanced training with mixed precision and checkpointing support"""
        self.neural_model.train()
        optimizer = torch.optim.AdamW(
            self.neural_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Create mixed precision trainer
        trainer = MixedPrecisionTrainer(
            model=self.neural_model,
            optimizer=optimizer,
            use_mixed_precision=self.use_mixed_precision,
            use_checkpointing=True
        )
        
        # Set up learning rate scheduler
        if scheduler == 'cosine':
            scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs * len(dataloader)
            )
        elif scheduler == 'reduce':
            scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2
            )
        else:
            scheduler_obj = None
            
        criterion = nn.CrossEntropyLoss()
        
        # For profiling
        if self.profiler:
            self.profiler.reset()
            
        # Training loop
        epoch_stats = []
        for epoch in range(num_epochs):
            if self.profiler:
                self.profiler.start_timer(f"epoch_{epoch+1}")
                
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Track progress
            if hasattr(dataloader, 'dataset'):
                progress_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                progress_iter = dataloader
                
            for inputs, labels in progress_iter:
                if self.profiler:
                    self.profiler.start_timer("batch_processing")
                    
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Perform training step
                loss = trainer.training_step(inputs, labels, criterion)
                
                # Update scheduler if using cosine
                if scheduler == 'cosine' and scheduler_obj is not None:
                    scheduler_obj.step()
                    
                # Calculate accuracy
                with torch.no_grad():
                    outputs, _, _ = self.neural_model(inputs, use_checkpoint=False)
                    _, predicted = torch.max(outputs.data, 1)
                    batch_total = labels.size(0)
                    batch_correct = (predicted == labels).sum().item()
                
                total += batch_total
                correct += batch_correct
                epoch_loss += loss * batch_total
                
                if self.profiler:
                    self.profiler.stop_timer("batch_processing")
                    
                # Print progress
                if hasattr(progress_iter, 'set_postfix'):
                    progress_iter.set_postfix({
                        'loss': f"{loss:.4f}",
                        'acc': f"{100 * correct / total:.2f}%"
                    })
                    
            # Calculate epoch statistics
            avg_loss = epoch_loss / total
            accuracy = 100 * correct / total
            
            # Update reduce-on-plateau scheduler if used
            if scheduler == 'reduce' and scheduler_obj is not None:
                scheduler_obj.step(avg_loss)
                
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Store epoch stats
            epoch_stats.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
            if self.profiler:
                self.profiler.stop_timer(f"epoch_{epoch+1}")
                
        # Print profiling summary if enabled
        if self.profiler:
            self.profiler.print_summary()
            
        return epoch_stats
    
    def diagnose(self, x, active_symptoms=None, risk_level='medium'):
        """
        Enhanced diagnosis with support for asynchronous reasoning and model parallelism
        """
        # Use parallel executor for diagnosis
        return self.parallel_executor.diagnose(
            x=x,
            active_symptoms=active_symptoms,
            
            risk_level=risk_level,
            max_hops=3,
            wait_for_symbolic=True  # Default to synchronous for compatibility
        )
    
    def diagnose_async(self, x, active_symptoms=None, risk_level='medium'):
        """
        Asynchronous diagnosis that returns immediately with neural results
        and continues symbolic reasoning in the background
        """
        if not self.async_symbolic_reasoning:
            logger.warning("Async reasoning not enabled, falling back to synchronous")
            return self.diagnose(x, active_symptoms, risk_level)
            
        # Use parallel executor for asynchronous diagnosis
        return self.parallel_executor.diagnose(
            x=x,
            active_symptoms=active_symptoms,
            risk_level=risk_level,
            max_hops=3,
            wait_for_symbolic=False  # Don't wait for symbolic reasoning
        )
    
    def check_pending_diagnosis(self, request_id):
        """
        Check if a pending asynchronous diagnosis is ready
        
        Args:
            request_id: The ID of the pending request
            
        Returns:
            The updated diagnosis result if ready, None if still pending
        """
        if not self.async_symbolic_reasoning:
            logger.warning("Async reasoning not enabled")
            return None
            
        return self.parallel_executor.check_pending_result(request_id)
    
    def wait_for_pending_diagnoses(self, timeout=None):
        """
        Wait for all pending diagnoses to complete
        
        Args:
            timeout: Maximum time to wait (None for no limit)
            
        Returns:
            Dictionary mapping request_id to diagnosis result
        """
        if not self.async_symbolic_reasoning:
            logger.warning("Async reasoning not enabled")
            return {}
            
        return self.parallel_executor.wait_for_all_pending(timeout)
    
    def evaluate(self, dataloader, symptom_dict=None, feedback=True, use_async=False, num_workers=1):
        """Enhanced evaluation with support for parallel and asynchronous processing"""
        self.neural_model.eval()
        self.interface.eval()
        
        # Initialize results
        for key in self.eval_results:
            self.eval_results[key]['correct'] = 0
            self.eval_results[key]['total'] = 0
            self.eval_results[key]['predictions'] = []
            self.eval_results[key]['true_labels'] = []
            self.eval_results[key]['confidence'] = []
            
        self.eval_results['neural']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['symbolic']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['nexus']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        
        agreement_cases = {
            'all_correct': 0, 
            'all_wrong': 0, 
            'neural_only': 0, 
            'symbolic_only': 0, 
            'nexus_better': 0
        }
        
        self.case_details = []
        
        # For profiling
        if self.profiler:
            self.profiler.reset()
            self.profiler.start_timer("total_evaluation")
            
        # Create execution plan if using async
        if use_async and self.async_symbolic_reasoning:
            execution_plan = NEXUSParallelExecutionPlan(
                executor=self.parallel_executor,
                batch_size=dataloader.batch_size if hasattr(dataloader, 'batch_size') else 32,
                max_pending=100,
                max_workers=num_workers
            )
            
            # Process the dataset asynchronously
            results = execution_plan.process_dataset(dataloader, symptom_dict)
        else:
            # Traditional evaluation approach
            results = []
            
            # Track progress
            if hasattr(dataloader, '__len__'):
                progress_iter = tqdm(dataloader, desc="Evaluating")
            else:
                progress_iter = dataloader
                
            sample_index = 0
            for i, (inputs, labels) in enumerate(progress_iter):
                if self.profiler:
                    self.profiler.start_timer("batch_evaluation")
                    
                batch_size = inputs.size(0)
                batch_results = []
                
                for j in range(batch_size):
                    # Get active symptoms for this sample if available
                    active_symptoms = symptom_dict.get(sample_index, None) if symptom_dict else None
                    
                    # Get diagnosis
                    sample_input = inputs[j].unsqueeze(0)
                    try:
                        with torch.no_grad():  # Add this to disable gradients during evaluation
                            result = self.diagnose(sample_input, active_symptoms)
                        batch_results.append(result)
                    except Exception as e:
                        print(f"Error processing sample {sample_index}: {str(e)}")
                        logger.error(f"Error in sample {sample_index}: {str(e)}")
                        batch_results.append({'error': str(e)})  # Track errors in results
                    
                    sample_index += 1
                    
                results.extend(batch_results)
                
                if self.profiler:
                    self.profiler.stop_timer("batch_evaluation")
        
        # Process results
        for i, result in enumerate(results):
            if self.profiler:
                self.profiler.start_timer("result_processing")
                
            # Check if result contains error
            if isinstance(result, dict) and 'error' in result:
                logger.error(f"Error in sample {i}: {result['error']}")
                continue
                
            # Extract true label
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__getitem__'):
                _, true_label = dataloader.dataset[i]
                true_label = true_label.item() if isinstance(true_label, torch.Tensor) else true_label
            else:
                # Default to 0 if true label can't be determined
                true_label = 0
                logger.warning(f"Could not determine true label for sample {i}")
            
            # Create case details
            case_detail = {
                'index': i,
                'true_label': true_label,
                'true_class': self.class_names[true_label] if true_label < len(self.class_names) else f"Unknown ({true_label})",
                'neural_pred': result['neural']['prediction'],
                'neural_conf': result['neural']['confidence'],
                'symbolic_pred': result['symbolic']['prediction'],
                'symbolic_conf': result['symbolic']['confidence'],
                'nexus_pred': result['nexus']['prediction'],
                'nexus_conf': result['nexus']['confidence'],
                'nexus_strategy': result['nexus']['strategy']['strategy'],
                'active_symptoms': result['symbolic'].get('active_symptoms', [])
            }
            self.case_details.append(case_detail)
            
            # Update evaluation metrics
            for key in ['neural', 'symbolic', 'nexus']:
                pred = result[key]['prediction']
                conf = result[key]['confidence']
                
                self.eval_results[key]['confusion'][true_label, pred] += 1
                self.eval_results[key]['predictions'].append(pred)
                self.eval_results[key]['true_labels'].append(true_label)
                self.eval_results[key]['confidence'].append(conf)
                
                if pred == true_label:
                    self.eval_results[key]['correct'] += 1
                self.eval_results[key]['total'] += 1
            
            # Check agreement between components
            neural_correct = result['neural']['prediction'] == true_label
            symbolic_correct = result['symbolic']['prediction'] == true_label
            nexus_correct = result['nexus']['prediction'] == true_label
            
            if neural_correct and symbolic_correct and nexus_correct:
                agreement_cases['all_correct'] += 1
            elif not neural_correct and not symbolic_correct and not nexus_correct:
                agreement_cases['all_wrong'] += 1
            elif neural_correct and not symbolic_correct:
                agreement_cases['neural_only'] += 1
            elif not neural_correct and symbolic_correct:
                agreement_cases['symbolic_only'] += 1
            elif nexus_correct and (not neural_correct or not symbolic_correct):
                agreement_cases['nexus_better'] += 1
            
            # Update metacognitive controller if feedback is enabled
            if feedback:
                self.metacognitive.update_thresholds(
                    neural_correct, 
                    symbolic_correct,
                    result['nexus']['strategy']['strategy']
                )
                
            if self.profiler:
                self.profiler.stop_timer("result_processing")
        
        # Calculate final metrics
        for key in self.eval_results:
            if self.eval_results[key]['total'] > 0:
                self.eval_results[key]['accuracy'] = (
                    self.eval_results[key]['correct'] / self.eval_results[key]['total']
                )
            else:
                self.eval_results[key]['accuracy'] = 0
                
        self.eval_results['agreement_cases'] = agreement_cases
        
        if self.profiler:
            self.profiler.stop_timer("total_evaluation")
            self.profiler.print_summary()
            
        return self.eval_results
    
    def explain_diagnosis(self, result, detail_level='medium', include_confidence=True):
        """Generate explanation for diagnosis results"""
        # Reuse original implementation
        from nexus_real_data import EnhancedNEXUSModel
        temp_model = EnhancedNEXUSModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            num_symbols=self.num_symbols,
            symbol_names=self.symbol_names,
            class_names=self.class_names
        )
        return temp_model.explain_diagnosis(result, detail_level, include_confidence)
    
    def visualize_results(self, output_prefix=None, save_figures=False, show_figures=True):
        """Generate visualizations for evaluation results"""
        # Reuse original implementation
        from nexus_real_data import EnhancedNEXUSModel
        temp_model = EnhancedNEXUSModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            num_symbols=self.num_symbols,
            symbol_names=self.symbol_names,
            class_names=self.class_names
        )
        temp_model.eval_results = self.eval_results
        return temp_model.visualize_results(output_prefix, save_figures, show_figures)
    
    def export_results(self, filename):
        """Export evaluation results to file"""
        # Reuse original implementation
        from nexus_real_data import EnhancedNEXUSModel
        temp_model = EnhancedNEXUSModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            num_symbols=self.num_symbols,
            symbol_names=self.symbol_names,
            class_names=self.class_names
        )
        temp_model.case_details = self.case_details
        return temp_model.export_results(filename)
    
    def save_model(self, directory):
        """Save the model and all its components"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'num_symbols': self.num_symbols,
            'embed_dim': self.embed_dim,
            'use_mixed_precision': self.use_mixed_precision,
            'async_symbolic_reasoning': self.async_symbolic_reasoning,
            'model_parallel': self.model_parallel,
            'num_gpus': self.num_gpus
        }
        
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            import json
            json.dump(config, f)
            
        # Save symbol and class names
        with open(os.path.join(directory, 'names.json'), 'w') as f:
            import json
            json.dump({
                'symbol_names': self.symbol_names,
                'class_names': self.class_names
            }, f)
            
        # Save neural model weights
        torch.save(
            self.neural_model.state_dict(),
            os.path.join(directory, 'neural_model.pt')
        )
        
        # Save interface weights
        torch.save(
            self.interface.state_dict(),
            os.path.join(directory, 'interface.pt')
        )
        
        # Save knowledge graph
        kg_dir = os.path.join(directory, 'knowledge_graph')
        self.knowledge_graph.save_to_disk(kg_dir)
        
        logger.info(f"Model saved to {directory}")
        
    @classmethod
    def load_model(cls, directory, device="cpu"):
        """Load model from saved directory"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        # Load configuration
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            import json
            config = json.load(f)
            
        # Load names
        with open(os.path.join(directory, 'names.json'), 'r') as f:
            names = json.load(f)
            
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
            num_symbols=config['num_symbols'],
            symbol_names=names['symbol_names'],
            class_names=names['class_names'],
            embed_dim=config.get('embed_dim', 128),
            device=device,
            use_mixed_precision=config.get('use_mixed_precision', False),
            async_symbolic_reasoning=config.get('async_symbolic_reasoning', False),
            model_parallel=config.get('model_parallel', False),
            num_gpus=config.get('num_gpus', 1)
        )
        
        # Load neural model weights
        model.neural_model.load_state_dict(
            torch.load(os.path.join(directory, 'neural_model.pt'), map_location=device)
        )
        
        # Load interface weights
        model.interface.load_state_dict(
            torch.load(os.path.join(directory, 'interface.pt'), map_location=device)
        )
        
        # Load knowledge graph
        kg_dir = os.path.join(directory, 'knowledge_graph')
        if hasattr(model.knowledge_graph, 'load_from_disk'):
            model.knowledge_graph = model.knowledge_graph.__class__.load_from_disk(kg_dir)
            
        logger.info(f"Model loaded from {directory}")
        return model


def run_scaled_nexus_experiment(
    dataset_name,
    max_samples,
    num_epochs,
    batch_size,
    learning_rate,
    output_dir,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_mixed_precision=True,
    use_sparse_kg=True,
    distributed_kg=False,
    async_reasoning=True,
    model_parallel=False,
    num_gpus=1,
    random_state=42
):
    """
    Run a NEXUS experiment with scaling optimizations
    """
    import time
    from datasets import load_dataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader
    
    print("=" * 80)
    print(f"Scaled NEXUS Experiment with {max_samples} samples")
    print("=" * 80)
    
    # Set random seeds
    random.seed(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    start_time = time.time()
    
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split='train')
    if max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    # Dynamically detect feature and label columns for the Iris dataset
    data_dict = dataset[0]
    
    # Special handling for Iris dataset which has different column names
    if "Species" in data_dict:
        # This is the Iris dataset
        feature_keys = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        label_key = 'Species'
        
        # Extract features and labels
        features = np.array([[sample[key] for key in feature_keys] for sample in dataset])
        labels = np.array([sample[label_key] for sample in dataset])
        
        # Convert string labels to numeric
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([label_to_id[label] for label in labels], dtype=int)
    else:
        # Try to find standard feature and label columns
        feature_key = next((k for k in data_dict if k in ['features', 'feature', 'text', 'data']), None)
        label_key = next((k for k in data_dict if k in ['label', 'labels', 'target']), None)
        
        if not feature_key or not label_key:
            raise ValueError(f"Dataset must contain standard feature and label columns. Found: {list(data_dict.keys())}")
        
        # Extract features and labels
        features = [sample[feature_key] for sample in dataset]
        labels = np.array([sample[label_key] for sample in dataset], dtype=int)
        
        # Convert features to numpy array, handling potential nested structures
        if isinstance(features[0], (list, np.ndarray)):
            features = np.array(features)
        elif isinstance(features[0], dict):
            # If features are dictionaries, assume they need flattening or selection
            feature_values = [list(f.values()) for f in features]
            features = np.array(feature_values)
        else:
            raise ValueError("Features must be numerical arrays or dictionaries")
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    class MedicalDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    train_dataset = MedicalDataset(X_train, y_train)
    test_dataset = MedicalDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    num_symbols = input_dim  # One symbol per feature
    
    if "Species" in data_dict:
        # Use actual feature and class names for Iris dataset
        symbol_names = feature_keys
        class_names = unique_labels
    else:
        # Generate symbol and class names
        symbol_names = [f"feature_{i}" for i in range(num_symbols)]
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    # Rest of the function remains the same...
    
    # Create scaled NEXUS model
    print("\nInitializing Scaled NEXUS model...")
    model = ScalableNEXUSModel(
        input_dim=input_dim,
        num_classes=num_classes,
        num_symbols=num_symbols,
        symbol_names=symbol_names,
        class_names=class_names,
        embed_dim=128,
        device=device,
        use_mixed_precision=use_mixed_precision,
        use_sparse_knowledge_graph=use_sparse_kg,
        distributed_knowledge_graph=distributed_kg,
        knowledge_graph_storage_dir=os.path.join(output_dir, "knowledge_graph"),
        async_symbolic_reasoning=async_reasoning,
        model_parallel=model_parallel,
        num_gpus=num_gpus,
        profiling=True
    )
    
    # Initialize knowledge graph
    print("Initializing knowledge graph...")
    model.init_knowledge_graph()
    
    # Train model
    print("\nTraining neural component...")
    model.train_neural(
        train_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        scheduler='cosine'
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    with torch.no_grad():  # Add this to ensure no gradients are computed during evaluation
        test_results = model.evaluate(
            test_loader,
            symptom_dict=None,
            feedback=False,
            use_async=async_reasoning
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export results
    model.export_results(os.path.join(output_dir, "evaluation_results.csv"))
    
    # Save model
    model.save_model(os.path.join(output_dir, "model"))
    
    # Generate visualizations
    model.visualize_results(
        output_prefix=os.path.join(output_dir, "results"),
        save_figures=True,
        show_figures=False
    )
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Experiment completed in {(end_time - start_time) / 60:.2f} minutes")
    print("=" * 80)
    
    print("\nFinal Comparative Summary:")
    print("-" * 40)
    print(f"Neural Model Accuracy: {test_results['neural']['accuracy']*100:.2f}%")
    print(f"Symbolic Model Accuracy: {test_results['symbolic']['accuracy']*100:.2f}%")
    print(f"NEXUS Model Accuracy: {test_results['nexus']['accuracy']*100:.2f}%")
    
    # Calculate improvements
    neural_acc = test_results['neural']['accuracy']
    symbolic_acc = test_results['symbolic']['accuracy']
    nexus_acc = test_results['nexus']['accuracy']
    
    best_component = max(neural_acc, symbolic_acc)
    improvement = (nexus_acc - best_component) * 100
    
    print(f"\nNEXUS improvement over best component: {improvement:.2f}%")
    
    # Agreement analysis
    agreement = test_results['agreement_cases']
    total = sum(agreement.values())
    
    print("\nAgreement Analysis:")
    if total > 0:  # Add check to avoid division by zero
        print(f"All models correct: {agreement['all_correct']} cases ({100*agreement['all_correct']/total:.1f}%)")
        print(f"Neural only correct: {agreement['neural_only']} cases ({100*agreement['neural_only']/total:.1f}%)")
        print(f"Symbolic only correct: {agreement['symbolic_only']} cases ({100*agreement['symbolic_only']/total:.1f}%)")
        print(f"NEXUS better than components: {agreement['nexus_better']} cases ({100*agreement['nexus_better']/total:.1f}%)")
        print(f"All models wrong: {agreement['all_wrong']} cases ({100*agreement['all_wrong']/total:.1f}%)")
    else:
        print("No valid agreement statistics available (total cases: 0)")
    
    # Print metacognitive strategy usage
    strategy_stats = model.metacognitive.get_strategy_stats()
    
    print("\nMetacognitive Strategy Usage:")
    if 'neural' in strategy_stats:
        print(f"Neural strategy: {strategy_stats['neural']*100:.1f}%")
        print(f"Symbolic strategy: {strategy_stats['symbolic']*100:.1f}%")
        print(f"Hybrid strategy: {strategy_stats['hybrid']*100:.1f}%")
    
    print(f"\nResults saved to: {output_dir}")
    
    return {
        'model': model,
        'test_results': test_results,
        'train_time': end_time - start_time
    }