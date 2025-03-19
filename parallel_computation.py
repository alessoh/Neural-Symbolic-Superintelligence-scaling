import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import threading
import queue
import concurrent.futures
from collections import defaultdict  # Add this import
from contextlib import contextmanager

class NEXUSDistributedTrainer:
    """
    Distributed training manager for NEXUS models using PyTorch DistributedDataParallel
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        world_size: int = None,
        rank: int = None,
        local_rank: int = None,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        checkpoint_dir: str = None
    ):
        self.model = model
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        self.rank = rank or int(os.environ.get("RANK", 0))
        self.local_rank = local_rank or int(os.environ.get("LOCAL_RANK", 0))
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.checkpoint_dir = checkpoint_dir
        
        # Setup device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            if device == "cuda":
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device(device)
                
        self.distributed = self.world_size > 1
        self.ddp_model = None
        
        # Initialize process group if distributed
        if self.distributed and not dist.is_initialized():
            self._setup_distributed()
            
    def _setup_distributed(self):
        """Initialize the distributed process group"""
        print(f"Initializing process group: rank={self.rank}, world_size={self.world_size}")
        
        # Set the device
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size
        )
        
        print(f"Process group initialized: {dist.is_initialized()}")
        
    def prepare_model(self):
        """Move model to device and wrap with DDP if distributed"""
        self.model = self.model.to(self.device)
        
        if self.distributed:
            # Wrap model with DDP
            self.ddp_model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=self.find_unused_parameters
            )
            print(f"Model wrapped with DistributedDataParallel")
        else:
            self.ddp_model = self.model
            
        return self.ddp_model
            
    def prepare_dataloader(self, dataset, batch_size, shuffle=True, **kwargs):
        """Create a dataloader with distributed sampler if needed"""
        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                **kwargs
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **kwargs
            )
            
    def save_checkpoint(self, epoch, optimizer, scheduler=None, metrics=None, filename=None):
        """Save a checkpoint of the model"""
        if self.checkpoint_dir is None:
            return
            
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Only save on rank 0 to avoid conflicts
        if self.rank == 0:
            if filename is None:
                filename = f"checkpoint_epoch_{epoch}.pt"
                
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # Get state dict from wrapped model if using DDP
            if self.distributed:
                state_dict = self.ddp_model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
                
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics or {}
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved to {filepath}")
            
    def load_checkpoint(self, filepath, optimizer=None, scheduler=None):
        """Load a checkpoint"""
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return None
            
        # Map to CPU if not using GPU to avoid GPU memory issues
        map_location = self.device if self.device.type == "cuda" else "cpu"
        
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load model state
        if self.distributed:
            self.ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint
        
    def all_reduce_scalar(self, value, op=dist.ReduceOp.SUM):
        """Perform all-reduce operation on a scalar value"""
        if not self.distributed:
            return value
            
        tensor = torch.tensor([value], device=self.device)
        dist.all_reduce(tensor, op=op)
        
        if op == dist.ReduceOp.SUM:
            return tensor.item() / self.world_size
        else:
            return tensor.item()
            
    def all_gather_object(self, obj):
        """Gather objects from all processes"""
        if not self.distributed:
            return [obj]
            
        gathered_objects = [None] * self.world_size
        dist.all_gather_object(gathered_objects, obj)
        return gathered_objects
        
    def barrier(self):
        """Synchronize all processes"""
        if self.distributed:
            dist.barrier()
            
    def cleanup(self):
        """Clean up the distributed environment"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


class ModelParallelNEXUS:
    """
    Model parallelism implementation for extremely large NEXUS models.
    This splits the transformer layers across multiple devices.
    """
    def __init__(
        self,
        neural_model,
        device_map=None,
        pipeline_chunks=1
    ):
        self.neural_model = neural_model
        self.device_map = device_map
        self.pipeline_chunks = pipeline_chunks
        
        # Apply device map if provided
        if self.device_map is not None:
            self._distribute_model()
            
    def _distribute_model(self):
        """Distribute model across devices according to device map"""
        # Example:
        # device_map = {
        #    'embedding': 0,
        #    'transformer_layers.0': 0,
        #    'transformer_layers.1': 1,
        #    'transformer_layers.2': 2,
        #    'classifier': 3
        # }
        
        for module_name, device_idx in self.device_map.items():
            device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
            
            try:
                module = self.neural_model
                for name in module_name.split('.'):
                    if name.isdigit():
                        module = module[int(name)]
                    else:
                        module = getattr(module, name)
                        
                module.to(device)
                print(f"Moved {module_name} to {device}")
            except AttributeError:
                print(f"Module not found: {module_name}")
                
    def _chunk_inputs(self, x, num_chunks):
        """Split input tensor into chunks for pipeline parallelism"""
        batch_size = x.size(0)
        chunk_size = batch_size // num_chunks
        
        if batch_size % num_chunks != 0:
            # Pad to make it divisible
            pad_size = num_chunks - (batch_size % num_chunks)
            x = torch.cat([x, torch.zeros(pad_size, *x.size()[1:], device=x.device)])
            chunk_size = (batch_size + pad_size) // num_chunks
            
        return x.split(chunk_size), batch_size
        
    def forward(self, x, use_pipeline=True):
        """
        Forward pass with model parallelism and optional pipeline parallelism
        
        Args:
            x: Input tensor
            use_pipeline: Whether to use pipeline parallelism
            
        Returns:
            Model outputs
        """
        if not use_pipeline or self.pipeline_chunks <= 1:
            # Standard model-parallel forward pass
            # The tensors will automatically move between devices
            return self.neural_model(x)
            
        # Pipeline parallelism with chunked inputs
        chunks, original_batch_size = self._chunk_inputs(x, self.pipeline_chunks)
        outputs = []
        
        # Process each chunk
        for chunk in chunks:
            # Forward pass for this chunk
            chunk_output = self.neural_model(chunk)
            outputs.append(chunk_output)
            
        # Combine outputs
        combined_outputs = torch.cat(outputs)
        
        # Remove padding if added
        if combined_outputs.size(0) > original_batch_size:
            combined_outputs = combined_outputs[:original_batch_size]
            
        return combined_outputs


class AsyncSymbolicReasoner:
    """
    Asynchronous symbolic reasoning engine that runs in parallel with neural computation
    """
    def __init__(self, knowledge_graph, num_workers=1):
        self.knowledge_graph = knowledge_graph
        self.num_workers = num_workers
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        
    def start(self):
        """Start the worker threads"""
        if self.running:
            return
            
        self.running = True
        
        # Create worker threads
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        print(f"Started {self.num_workers} async symbolic reasoning workers")
        
    def stop(self):
        """Stop the worker threads"""
        if not self.running:
            return
            
        self.running = False
        
        # Put sentinel values to stop workers
        for _ in range(self.num_workers):
            self.request_queue.put(None)
            
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
            
        self.workers = []
        print("Stopped all async symbolic reasoning workers")
        
    def _worker_loop(self):
        """Worker thread loop"""
        while self.running:
            # Get a request from the queue
            request = self.request_queue.get()
            
            # Check for sentinel value
            if request is None:
                self.request_queue.task_done()
                break
                
            try:
                # Process the request
                request_id, active_entities, max_hops = request
                
                # Perform reasoning
                result = self.knowledge_graph.reason(active_entities, max_hops)
                
                # Put result in the result queue
                self.result_queue.put((request_id, result))
            except Exception as e:
                # Put error in result queue
                self.result_queue.put((request_id, e))
            finally:
                # Mark task as done
                self.request_queue.task_done()
                
    def submit_reasoning_task(self, request_id, active_entities, max_hops=3):
        """
        Submit a reasoning task to be processed asynchronously
        
        Args:
            request_id: Unique identifier for this request
            active_entities: List of active entity IDs
            max_hops: Maximum number of reasoning hops
            
        Returns:
            request_id for tracking the result
        """
        if not self.running:
            self.start()
            
        self.request_queue.put((request_id, active_entities, max_hops))
        return request_id
        
    def get_result(self, timeout=None):
        """
        Get a result from the result queue
        
        Args:
            timeout: How long to wait for a result (None means wait indefinitely)
            
        Returns:
            (request_id, result) tuple or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_result_for_request(self, request_id, timeout=None):
        """
        Get the result for a specific request ID
        
        Args:
            request_id: The request ID to look for
            timeout: Total time to wait before giving up
            
        Returns:
            Result for this request or None if not found within timeout
        """
        start_time = time.time()
        results = []
        
        while timeout is None or (time.time() - start_time) < timeout:
            try:
                result = self.result_queue.get(timeout=0.1)
                
                if result[0] == request_id:
                    # Put back any other results we retrieved
                    for r in results:
                        self.result_queue.put(r)
                    return result[1]
                else:
                    results.append(result)
            except queue.Empty:
                pass
                
        # Put back all results
        for r in results:
            self.result_queue.put(r)
            
        return None
        

class NEXUSParallelExecutor:
    """
    Orchestrates parallel execution of neural and symbolic components
    """
    def __init__(
        self,
        neural_model,
        knowledge_graph,
        device="cuda",
        num_symbolic_workers=1,
        enable_async=True
    ):
        self.neural_model = neural_model
        self.knowledge_graph = knowledge_graph
        self.device = device
        self.enable_async = enable_async
        
        # Move neural model to device
        if device != "cpu":
            self.neural_model = self.neural_model.to(device)
            
        # Create symbolic reasoner
        if enable_async:
            self.symbolic_reasoner = AsyncSymbolicReasoner(
                knowledge_graph,
                num_workers=num_symbolic_workers
            )
            self.symbolic_reasoner.start()
        else:
            self.symbolic_reasoner = None
            
        # Request tracking
        self.next_request_id = 0
        self.pending_requests = {}
        
    def __del__(self):
        """Clean up resources"""
        if self.symbolic_reasoner is not None:
            self.symbolic_reasoner.stop()
            
    def get_next_request_id(self):
        """Get a unique request ID"""
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id
        
    @contextmanager
    def neural_computation(self):
        """Context manager for neural computation phase"""
        # Device synchronization if needed
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.time()
        yield
        
        # Device synchronization after computation
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        duration = time.time() - start_time
        return duration
        
    def diagnose(self, x, active_symptoms=None, risk_level='medium', max_hops=3, wait_for_symbolic=True):
        """
        Asynchronous diagnosis using parallel neural and symbolic computation
        
        Args:
            x: Input features
            active_symptoms: Optional list of active symptoms
            risk_level: Risk level for metacognitive controller
            max_hops: Maximum reasoning hops
            wait_for_symbolic: Whether to wait for symbolic reasoning to complete
            
        Returns:
            Diagnosis result
        """
        # Move input to device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        # Start neural computation
        with self.neural_computation():
            # Run neural model forward pass
            neural_logits, neural_repr, _ = self.neural_model(x)
            neural_probs = F.softmax(neural_logits, dim=1)
            neural_pred = torch.argmax(neural_probs, dim=1).item()
            neural_conf = neural_probs[0, neural_pred].item()
            
            # Neural to symbolic translation
            if hasattr(self.neural_model, 'interface'):
                symbolic_activations, similarities, _ = self.neural_model.interface.neural_to_symbolic(neural_repr)
            else:
                # Default method for models without interface
                symbolic_activations = torch.sigmoid(neural_repr) > 0.5
                similarities = torch.sigmoid(neural_repr)
        
        # Prepare for symbolic reasoning
        if active_symptoms is not None:
            # Use provided symptoms
            if hasattr(self.neural_model, 'symbol_to_id'):
                symptom_ids = [self.neural_model.symbol_to_id[name] for name in active_symptoms 
                             if name in self.neural_model.symbol_to_id]
            else:
                # Default handling if symbol_to_id is not available
                raise ValueError("Model lacks symbol_to_id mapping needed for active_symptoms")
        else:
            # Extract activated symptoms from neural representations
            symptom_ids = torch.nonzero(symbolic_activations[0]).squeeze(-1).tolist()
            if not isinstance(symptom_ids, list):
                symptom_ids = [symptom_ids]
        
        # Get symbol and class names for result creation
        symbol_names = self.neural_model.symbol_names if hasattr(self.neural_model, 'symbol_names') else [f"symbol_{i}" for i in range(self.neural_model.num_symbols)]
        class_names = self.neural_model.class_names if hasattr(self.neural_model, 'class_names') else [f"Class {i}" for i in range(self.neural_model.num_classes)]
        
        # Start symbolic reasoning
        if self.enable_async and not wait_for_symbolic:
            # Asynchronous reasoning
            request_id = self.get_next_request_id()
            self.symbolic_reasoner.submit_reasoning_task(request_id, symptom_ids, max_hops)
            
            # Create initial result with neural components and pending symbolic
            # Use placeholder values for symbolic prediction
            symbolic_pred = neural_pred  # Default to neural prediction for now
            symbolic_conf = 0.0  # Unknown confidence
            
            # Create result with pending symbolic computation
            result = {
                'neural': {
                    'prediction': neural_pred,
                    'confidence': neural_conf,
                    'class_name': class_names[neural_pred],
                    'probabilities': neural_probs[0].detach().cpu().numpy()
                },
                'symbolic': {
                    'prediction': symbolic_pred,
                    'confidence': symbolic_conf,
                    'class_name': class_names[symbolic_pred],
                    'reasoning_steps': {},  # Will be filled later
                    'inferred_symbols': [],  # Will be filled later
                    'active_symptoms': [symbol_names[i] for i in symptom_ids if i < len(symbol_names)],
                    'class_scores': {},  # Will be filled later
                    'probabilities': torch.zeros(self.neural_model.num_classes).cpu().numpy(),  # Placeholder
                    'request_id': request_id,
                    'pending': True
                },
                'nexus': {
                    'prediction': neural_pred,  # Use neural prediction for now
                    'confidence': neural_conf,
                    'class_name': class_names[neural_pred],
                    'strategy': {'strategy': 'neural', 'explanation': 'Initial neural prediction while waiting for symbolic reasoning'}
                }
            }
            
            # Store pending request
            self.pending_requests[request_id] = result
            
            return result
        else:
            # Synchronous reasoning
            if self.enable_async:
                # Use the async reasoner but wait for result
                request_id = self.get_next_request_id()
                self.symbolic_reasoner.submit_reasoning_task(request_id, symptom_ids, max_hops)
                symbolic_result = self.symbolic_reasoner.get_result_for_request(request_id)
                if symbolic_result is None:
                    # Fallback if async result not available
                    inferred, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids, max_hops)
                else:
                    inferred, reasoning_steps, confidences, class_scores = symbolic_result
            else:
                # Direct reasoning
                inferred, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids, max_hops)
            
            # Process symbolic results
            symbolic_scores = torch.zeros(1, self.neural_model.num_classes, device=self.device)
            for class_id, score in class_scores.items():
                if class_id < self.neural_model.num_classes:
                    symbolic_scores[0, class_id] = score
                    
            if symbolic_scores.sum() == 0:
                symbolic_probs = torch.ones(1, self.neural_model.num_classes, device=self.device) / self.neural_model.num_classes
            else:
                symbolic_probs = F.softmax(symbolic_scores, dim=1)
                
            symbolic_pred = torch.argmax(symbolic_probs, dim=1).item()
            symbolic_conf = symbolic_probs[0, symbolic_pred].item()
            
            # Metacognitive control
            if hasattr(self.neural_model, 'metacognitive'):
                strategy = self.neural_model.metacognitive.decide_strategy(neural_conf, symbolic_conf, risk_level)
            else:
                # Default strategy if metacognitive controller not available
                if neural_conf >= 0.8 and symbolic_conf < 0.7:
                    strategy = {
                        'strategy': 'neural',
                        'neural_weight': 1.0,
                        'symbolic_weight': 0.0,
                        'explanation': f'Using neural prediction (high confidence: {neural_conf:.2f})'
                    }
                elif symbolic_conf >= 0.7 and neural_conf < 0.8:
                    strategy = {
                        'strategy': 'symbolic',
                        'neural_weight': 0.0,
                        'symbolic_weight': 1.0,
                        'explanation': f'Using symbolic reasoning (high confidence: {symbolic_conf:.2f})'
                    }
                else:
                    total_conf = neural_conf + symbolic_conf
                    neural_weight = neural_conf / total_conf if total_conf > 0 else 0.5
                    symbolic_weight = 1.0 - neural_weight
                    strategy = {
                        'strategy': 'hybrid',
                        'neural_weight': neural_weight,
                        'symbolic_weight': symbolic_weight,
                        'explanation': f'Using weighted combination (neural: {neural_weight:.2f}, symbolic: {symbolic_weight:.2f})'
                    }
            
            # Final prediction based on strategy
            if strategy['strategy'] == 'neural':
                final_pred = neural_pred
                final_conf = neural_conf
            elif strategy['strategy'] == 'symbolic':
                final_pred = symbolic_pred
                final_conf = symbolic_conf
            else:  # hybrid
                combined_probs = (
                    strategy['neural_weight'] * neural_probs + 
                    strategy['symbolic_weight'] * symbolic_probs
                )
                final_pred = torch.argmax(combined_probs, dim=1).item()
                final_conf = combined_probs[0, final_pred].item()
            
            # Create complete result
            result = {
                'neural': {
                    'prediction': neural_pred,
                    'confidence': neural_conf,
                    'class_name': class_names[neural_pred],
                    'probabilities': neural_probs[0].cpu().numpy()
                },
                'symbolic': {
                    'prediction': symbolic_pred,
                    'confidence': symbolic_conf,
                    'class_name': class_names[symbolic_pred],
                    'reasoning_steps': reasoning_steps,
                    'inferred_symbols': [symbol_names[i] for i in inferred if i < len(symbol_names)],
                    'active_symptoms': [symbol_names[i] for i in symptom_ids if i < len(symbol_names)],
                    'class_scores': class_scores,
                    'probabilities': symbolic_probs[0].cpu().numpy()
                },
                'nexus': {
                    'prediction': final_pred,
                    'confidence': final_conf,
                    'class_name': class_names[final_pred],
                    'strategy': strategy
                }
            }
            
            return result
            
    def check_pending_result(self, request_id):
        """
        Check if a pending reasoning result is ready
        
        Args:
            request_id: ID of the pending request
            
        Returns:
            Updated result if ready, None if still pending
        """
        if not self.enable_async or request_id not in self.pending_requests:
            return None
            
        # Try to get the result
        symbolic_result = self.symbolic_reasoner.get_result_for_request(request_id, timeout=0)
        
        if symbolic_result is None:
            # Still pending
            return None
            
        # Result is ready, update the pending result
        result = self.pending_requests[request_id]
        
        # Unpack symbolic results
        inferred, reasoning_steps, confidences, class_scores = symbolic_result
        
        # Process symbolic results
        symbolic_scores = torch.zeros(1, self.neural_model.num_classes, device=self.device)
        for class_id, score in class_scores.items():
            if class_id < self.neural_model.num_classes:
                symbolic_scores[0, class_id] = score
                
        if symbolic_scores.sum() == 0:
            symbolic_probs = torch.ones(1, self.neural_model.num_classes, device=self.device) / self.neural_model.num_classes
        else:
            symbolic_probs = F.softmax(symbolic_scores, dim=1)
            
        symbolic_pred = torch.argmax(symbolic_probs, dim=1).item()
        symbolic_conf = symbolic_probs[0, symbolic_pred].item()
        
        # Get neural results from the saved pending result
        neural_pred = result['neural']['prediction']
        neural_conf = result['neural']['confidence']
        
        # Metacognitive control
        if hasattr(self.neural_model, 'metacognitive'):
            strategy = self.neural_model.metacognitive.decide_strategy(neural_conf, symbolic_conf)
        else:
            # Default strategy
            if neural_conf >= 0.8 and symbolic_conf < 0.7:
                strategy = {
                    'strategy': 'neural',
                    'neural_weight': 1.0,
                    'symbolic_weight': 0.0,
                    'explanation': f'Using neural prediction (high confidence: {neural_conf:.2f})'
                }
            elif symbolic_conf >= 0.7 and neural_conf < 0.8:
                strategy = {
                    'strategy': 'symbolic',
                    'neural_weight': 0.0,
                    'symbolic_weight': 1.0,
                    'explanation': f'Using symbolic reasoning (high confidence: {symbolic_conf:.2f})'
                }
            else:
                total_conf = neural_conf + symbolic_conf
                neural_weight = neural_conf / total_conf if total_conf > 0 else 0.5
                symbolic_weight = 1.0 - neural_weight
                strategy = {
                    'strategy': 'hybrid',
                    'neural_weight': neural_weight,
                    'symbolic_weight': symbolic_weight,
                    'explanation': f'Using weighted combination (neural: {neural_weight:.2f}, symbolic: {symbolic_weight:.2f})'
                }
        
        # Update with final prediction
        if strategy['strategy'] == 'neural':
            final_pred = neural_pred
            final_conf = neural_conf
        elif strategy['strategy'] == 'symbolic':
            final_pred = symbolic_pred
            final_conf = symbolic_conf
        else:  # hybrid
            neural_probs = torch.tensor(result['neural']['probabilities'], device=self.device).unsqueeze(0)
            combined_probs = (
                strategy['neural_weight'] * neural_probs + 
                strategy['symbolic_weight'] * symbolic_probs
            )
            final_pred = torch.argmax(combined_probs, dim=1).item()
            final_conf = combined_probs[0, final_pred].item()
        
        # Get symbol and class names
        symbol_names = self.neural_model.symbol_names if hasattr(self.neural_model, 'symbol_names') else [f"symbol_{i}" for i in range(self.neural_model.num_symbols)]
        class_names = self.neural_model.class_names if hasattr(self.neural_model, 'class_names') else [f"Class {i}" for i in range(self.neural_model.num_classes)]
        
        # Update the result
        result['symbolic'] = {
            'prediction': symbolic_pred,
            'confidence': symbolic_conf,
            'class_name': class_names[symbolic_pred],
            'reasoning_steps': reasoning_steps,
            'inferred_symbols': [symbol_names[i] for i in inferred if i < len(symbol_names)],
            'active_symptoms': result.get('symbolic', {}).get('active_symptoms', []),
            'class_scores': class_scores,
            'probabilities': symbolic_probs[0].cpu().numpy(),
            'pending': False
        }
        
        result['nexus'] = {
            'prediction': final_pred,
            'confidence': final_conf,
            'class_name': class_names[final_pred],
            'strategy': strategy}
        
        # Remove from pending requests
        del self.pending_requests[request_id]
        
        return result
    
    def wait_for_all_pending(self, timeout=None):
        """
        Wait for all pending reasoning tasks to complete
        
        Args:
            timeout: Maximum time to wait (None for no limit)
            
        Returns:
            Dictionary mapping request_id to result
        """
        if not self.enable_async or not self.pending_requests:
            return {}
            
        start_time = time.time()
        results = {}
        pending_ids = list(self.pending_requests.keys())
        
        while pending_ids and (timeout is None or (time.time() - start_time) < timeout):
            # Check each pending request
            for request_id in list(pending_ids):
                result = self.check_pending_result(request_id)
                if result is not None:
                    results[request_id] = result
                    pending_ids.remove(request_id)
                    
            # Small sleep to avoid busy waiting
            if pending_ids:
                time.sleep(0.01)
                
        return results


class NEXUSParallelExecutionPlan:
    """
    Advanced execution plan for batch processing with NEXUS
    that optimizes parallel execution of neural and symbolic components
    """
    def __init__(
        self,
        executor: NEXUSParallelExecutor,
        batch_size: int = 32,
        max_pending: int = 100,
        max_workers: int = 4
    ):
        self.executor = executor
        self.batch_size = batch_size
        self.max_pending = max_pending
        
        # Thread pool for parallel execution
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        
    def process_batch(self, inputs, active_symptoms_list=None, risk_levels=None):
        """
        Process a batch of inputs with parallel execution
        
        Args:
            inputs: List of input tensors
            active_symptoms_list: Optional list of active symptoms for each input
            risk_levels: Optional list of risk levels for each input
            
        Returns:
            List of results
        """
        batch_size = len(inputs)
        futures = []
        
        # Default values
        if active_symptoms_list is None:
            active_symptoms_list = [None] * batch_size
            
        if risk_levels is None:
            risk_levels = ['medium'] * batch_size
            
        # Submit tasks
        for i in range(batch_size):
            future = self.thread_pool.submit(
                self.executor.diagnose,
                inputs[i],
                active_symptoms_list[i],
                risk_levels[i],
                3,  # max_hops
                True  # wait_for_symbolic
            )
            futures.append(future)
            
        # Wait for all tasks to complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle exceptions
                print(f"Error in batch processing: {e}")
                results.append({'error': str(e)})
                
        return results
        
    def process_dataset(self, dataloader, active_symptoms_dict=None, risk_level_dict=None):
        """
        Process an entire dataset with optimized parallel execution
        
        Args:
            dataloader: PyTorch DataLoader
            active_symptoms_dict: Optional dictionary mapping indices to active symptoms
            risk_level_dict: Optional dictionary mapping indices to risk levels
            
        Returns:
            List of results
        """
        all_results = []
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            batch_results = []
            
            # Process each sample in the batch
            for i in range(inputs.size(0)):
                sample_idx = batch_idx * dataloader.batch_size + i
                
                # Get active symptoms and risk level if provided
                active_symptoms = None
                if active_symptoms_dict and sample_idx in active_symptoms_dict:
                    active_symptoms = active_symptoms_dict[sample_idx]
                    
                risk_level = 'medium'
                if risk_level_dict and sample_idx in risk_level_dict:
                    risk_level = risk_level_dict[sample_idx]
                    
                # Submit the task
                future = self.thread_pool.submit(
                    self.executor.diagnose,
                    inputs[i],
                    active_symptoms,
                    risk_level,
                    3,  # max_hops
                    False  # don't wait for symbolic
                )
                
                # Store the future with sample index
                self.futures[sample_idx] = future
                
                # Check if we should wait for some futures
                while len(self.futures) >= self.max_pending:
                    # Wait for at least one future to complete
                    completed, _ = concurrent.futures.wait(
                        list(self.futures.values()),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # Process completed futures
                    for future in completed:
                        for idx, f in list(self.futures.items()):
                            if f == future:
                                try:
                                    result = future.result()
                                    
                                    # If result has pending symbolic reasoning, wait for it
                                    if 'symbolic' in result and result['symbolic'].get('pending', False):
                                        request_id = result['symbolic']['request_id']
                                        complete_result = self.executor.check_pending_result(request_id)
                                        if complete_result is not None:
                                            result = complete_result
                                            
                                    all_results.append((idx, result))
                                    del self.futures[idx]
                                except Exception as e:
                                    print(f"Error processing sample {idx}: {e}")
                                    all_results.append((idx, {'error': str(e)}))
                                    del self.futures[idx]
                                break
            
        # Wait for remaining futures
        while self.futures:
            # Wait for the next future to complete
            completed, _ = concurrent.futures.wait(
                list(self.futures.values()),
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Process completed future
            for future in completed:
                for idx, f in list(self.futures.items()):
                    if f == future:
                        try:
                            result = future.result()
                            
                            # Wait for pending symbolic reasoning
                            if 'symbolic' in result and result['symbolic'].get('pending', False):
                                request_id = result['symbolic']['request_id']
                                complete_result = self.executor.wait_for_all_pending().get(request_id)
                                if complete_result is not None:
                                    result = complete_result
                                    
                            all_results.append((idx, result))
                            del self.futures[idx]
                        except Exception as e:
                            print(f"Error processing sample {idx}: {e}")
                            all_results.append((idx, {'error': str(e)}))
                            del self.futures[idx]
                        break
        
        # Sort results by index
        all_results.sort(key=lambda x: x[0])
        return [r for _, r in all_results]


class NEXUSPerformanceProfiler:
    """
    Performance profiler for NEXUS to identify bottlenecks and optimize execution
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.start_times = {}
        
    def start_timer(self, name):
        """Start timing a specific operation"""
        self.start_times[name] = time.time()
        
    def stop_timer(self, name):
        """Stop timing and record the duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.timings[name].append(duration)
            self.counters[name] += 1
            del self.start_times[name]
            return duration
        return None
        
    def increment_counter(self, name, value=1):
        """Increment a counter"""
        self.counters[name] += value
        
    def get_average_time(self, name):
        """Get the average time for an operation"""
        if name in self.timings and self.timings[name]:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0
        
    def get_total_time(self, name):
        """Get the total time for an operation"""
        if name in self.timings:
            return sum(self.timings[name])
        return 0
        
    def get_count(self, name):
        """Get the count for an operation or counter"""
        return self.counters[name]
        
    def reset(self):
        """Reset all timings and counters"""
        self.timings.clear()
        self.counters.clear()
        self.start_times.clear()
        
    def summary(self):
        """Generate a performance summary"""
        result = {
            'timings': {},
            'counters': dict(self.counters)
        }
        
        for name, times in self.timings.items():
            if times:
                result['timings'][name] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
                
        return result
        
    def print_summary(self):
        """Print a formatted performance summary"""
        summary = self.summary()
        
        print("\n==== NEXUS Performance Summary ====")
        
        print("\nTimings (seconds):")
        headers = ["Operation", "Count", "Total", "Average", "Min", "Max"]
        rows = []
        
        for name, stats in summary['timings'].items():
            rows.append([
                name,
                stats['count'],
                f"{stats['total']:.4f}s",
                f"{stats['average']:.4f}s",
                f"{stats['min']:.4f}s",
                f"{stats['max']:.4f}s"
            ])
        
        # Sort by total time (descending)
        rows.sort(key=lambda x: float(x[2][:-1]), reverse=True)
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        # Print header
        header_str = " | ".join(f"{headers[i]:{col_widths[i]}}" for i in range(len(headers)))
        print(header_str)
        print("-" * len(header_str))
        
        # Print rows
        for row in rows:
            print(" | ".join(f"{row[i]:{col_widths[i] if i != 0 else str(col_widths[i])+'s'}}" for i in range(len(row))))
        
        print("\nCounters:")
        for name, count in sorted(summary['counters'].items(), key=lambda x: x[1], reverse=True):
            if name not in summary['timings']:  # Only show counters that aren't already in timings
                print(f"  {name}: {count}")

# Utility functions for parallel computation

def launch_distributed_training(rank, world_size, main_func, backend='nccl', **kwargs):
    """
    Utility function for launching distributed training on multiple GPUs
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        main_func: Main training function
        backend: Backend for distributed training
        **kwargs: Additional arguments to pass to main_func
        
    Example usage:
        # In main script
        def main_worker(rank, world_size, args):
            # Initialize distributed
            trainer = NEXUSDistributedTrainer(model, rank=rank, world_size=world_size)
            # Rest of training code...
            
        if __name__ == "__main__":
            torch.multiprocessing.spawn(
                launch_distributed_training,
                args=(4, main_worker, 'nccl', args),
                nprocs=4
            )
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # Call the main function
    main_func(rank, world_size, **kwargs)
    
def create_optimal_device_map(model, num_gpus):
    """
    Create an optimal device map for model parallelism
    
    Args:
        model: The model to distribute
        num_gpus: Number of available GPUs
        
    Returns:
        Dictionary mapping module names to device indices
    """
    if num_gpus <= 1:
        return None
        
    device_map = {}
    
    # Get all modules
    modules = []
    for name, _ in model.named_modules():
        if name and '.' not in name:  # Get top-level modules
            modules.append(name)
            
    # Add transformer layers specifically
    if hasattr(model, 'transformer_layers'):
        for i in range(len(model.transformer_layers)):
            modules.remove('transformer_layers')  # Remove the parent
            modules.append(f'transformer_layers.{i}')
            
    # Sort modules to keep related components together
    modules.sort()
    
    # Distribute modules across GPUs
    chunk_size = max(1, len(modules) // num_gpus)
    
    for i, module_name in enumerate(modules):
        device_idx = min(i // chunk_size, num_gpus - 1)
        device_map[module_name] = device_idx
        
    return device_map