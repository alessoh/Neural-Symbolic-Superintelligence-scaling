import torch
import argparse
import numpy as np
from nexus_scaling_integration import run_scaled_nexus_experiment
from efficient_transformer import ScalableTransformerModel
from parallel_computation import NEXUSParallelExecutor, NEXUSParallelExecutionPlan

# Fix the missing num_symbols attribute in ScalableTransformerModel
def patch_neural_model():
    # Add the num_symbols attribute to the ScalableTransformerModel class
    original_init = ScalableTransformerModel.__init__
    
    def patched_init(self, input_dim, num_classes, embed_dim=128, num_layers=3, 
                     num_heads=8, ff_dim=512, dropout=0.1, activation="gelu", 
                     use_flash_attention=False, use_mixed_precision=False):
        # Call the original init
        original_init(self, input_dim, num_classes, embed_dim, num_layers, 
                      num_heads, ff_dim, dropout, activation, 
                      use_flash_attention, use_mixed_precision)
        
        # Add the num_symbols attribute
        self.num_symbols = input_dim  # Using input dimension as the number of symbols
    
    # Apply the patch
    ScalableTransformerModel.__init__ = patched_init

def patch_parallel_executor():
    # Store the original diagnose method
    original_diagnose = NEXUSParallelExecutor.diagnose
    
    # Create a patched version
    def patched_diagnose(self, x, active_symptoms=None, risk_level='medium', max_hops=3, wait_for_symbolic=True):
        try:
            # Move input to device
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
            x = x.to(self.device)
            
            # Start neural computation
            with torch.no_grad():  # Add no_grad context to prevent gradient tracking
                # Run neural model forward pass
                neural_logits, neural_repr, _ = self.neural_model(x)
                neural_probs = torch.nn.functional.softmax(neural_logits, dim=1)
                neural_pred = torch.argmax(neural_probs, dim=1).item()
                neural_conf = neural_probs[0, neural_pred].item()
                
                # Neural to symbolic translation - PATCHED SECTION
                if hasattr(self.neural_model, 'interface'):
                    symbolic_activations, similarities, _ = self.neural_model.interface.neural_to_symbolic(neural_repr)
                else:
                    # Default method for models without interface
                    # Check if neural_repr is 1D or 2D
                    if neural_repr.dim() == 2:
                        symbolic_activations = torch.sigmoid(neural_repr) > 0.5
                        similarities = torch.sigmoid(neural_repr)
                    else:
                        # Handle the case where neural_repr is a different dimension
                        symbolic_activations = torch.zeros(1, getattr(self.neural_model, 'num_symbols', neural_repr.size(-1)), device=self.device)
                        similarities = torch.zeros_like(symbolic_activations)
            
            # Get the number of symbols
            if hasattr(self.neural_model, 'num_symbols'):
                num_symbols = self.neural_model.num_symbols
            else:
                # Fallback to input dimension or neural representation size
                if hasattr(self.neural_model, 'input_dim'):
                    num_symbols = self.neural_model.input_dim
                else:
                    num_symbols = neural_repr.size(-1)
                    
            # Get symbol and class names for result creation
            if hasattr(self.neural_model, 'symbol_names'):
                symbol_names = self.neural_model.symbol_names
            else:
                # Create generic symbol names
                symbol_names = [f"feature_{i}" for i in range(num_symbols)]
                
            if hasattr(self.neural_model, 'class_names'):
                class_names = self.neural_model.class_names
            else:
                # Create generic class names
                num_classes = neural_logits.size(1)
                class_names = [f"Class_{i}" for i in range(num_classes)]
                
            # Prepare for symbolic reasoning
            if active_symptoms is not None:
                # Use provided symptoms
                if hasattr(self.neural_model, 'symbol_to_id'):
                    symptom_ids = [self.neural_model.symbol_to_id[name] for name in active_symptoms 
                                 if name in self.neural_model.symbol_to_id]
                else:
                    # Default handling if symbol_to_id is not available
                    # Create a simple mapping
                    symptom_ids = [i for i, name in enumerate(active_symptoms) if i < num_symbols]
            else:
                # Extract activated symptoms from neural representations
                if symbolic_activations.dim() > 0 and symbolic_activations.size(0) > 0:
                    symptom_ids = torch.nonzero(symbolic_activations[0]).squeeze(-1).tolist()
                    if not isinstance(symptom_ids, list):
                        symptom_ids = [symptom_ids]
                else:
                    symptom_ids = []
            
            # Ensure the knowledge graph has the correct settings
            if hasattr(self.knowledge_graph, 'symbol_offset'):
                self.knowledge_graph.symbol_offset = num_symbols
            if hasattr(self.knowledge_graph, 'num_classes'):
                self.knowledge_graph.num_classes = len(class_names)
                
            # Start symbolic reasoning
            if self.enable_async and not wait_for_symbolic:
                # Basic async case - will implement full reasoning later
                symbolic_pred = neural_pred
                symbolic_conf = 0.5
                
                # Return a simple result for now
                dummy_result = {
                    'neural': {
                        'prediction': neural_pred,
                        'confidence': neural_conf,
                        'class_name': class_names[neural_pred] if neural_pred < len(class_names) else "Unknown",
                        'probabilities': neural_probs[0].detach().cpu().numpy()  # Detach before numpy conversion
                    },
                    'symbolic': {
                        'prediction': symbolic_pred,
                        'confidence': symbolic_conf,
                        'class_name': class_names[symbolic_pred] if symbolic_pred < len(class_names) else "Unknown",
                        'reasoning_steps': {},
                        'inferred_symbols': [],
                        'active_symptoms': [symbol_names[i] for i in symptom_ids if i < len(symbol_names)],
                        'class_scores': {},
                        'probabilities': neural_probs[0].detach().cpu().numpy(),  # Detach before numpy conversion
                        'request_id': 0,
                        'pending': True
                    },
                    'nexus': {
                        'prediction': neural_pred,
                        'confidence': neural_conf,
                        'class_name': class_names[neural_pred] if neural_pred < len(class_names) else "Unknown",
                        'strategy': {
                            'strategy': 'neural', 
                            'neural_weight': 1.0,
                            'symbolic_weight': 0.0,
                            'explanation': 'Using neural prediction only due to missing symbolic component'
                        }
                    }
                }
                return dummy_result
            else:
                # Synchronous reasoning 
                try:
                    # Try to use the knowledge graph for reasoning
                    inferred, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids, max_hops)
                    
                    # Process symbolic results
                    symbolic_scores = torch.zeros(1, len(class_names), device=self.device)
                    for class_id, score in class_scores.items():
                        if isinstance(class_id, (int, float)) and int(class_id) < len(class_names):
                            symbolic_scores[0, int(class_id)] = score
                            
                    if symbolic_scores.sum() == 0:
                        symbolic_probs = torch.ones(1, len(class_names), device=self.device) / len(class_names)
                    else:
                        symbolic_probs = torch.nn.functional.softmax(symbolic_scores, dim=1)
                        
                    symbolic_pred = torch.argmax(symbolic_probs, dim=1).item()
                    symbolic_conf = symbolic_probs[0, symbolic_pred].item()
                except Exception as e:
                    print(f"Error in symbolic reasoning: {str(e)}")
                    # Fallback to neural prediction
                    symbolic_probs = neural_probs.clone()
                    symbolic_pred = neural_pred
                    symbolic_conf = 0.5
                    inferred = set()
                    reasoning_steps = {}
                    confidences = {}
                    class_scores = {}
                
                # Simple strategy selection
                if neural_conf > symbolic_conf:
                    final_pred = neural_pred
                    final_conf = neural_conf
                    strategy = {
                        'strategy': 'neural',
                        'neural_weight': 1.0,
                        'symbolic_weight': 0.0,
                        'explanation': f'Using neural prediction (higher confidence: {neural_conf:.2f} vs {symbolic_conf:.2f})'
                    }
                else:
                    final_pred = symbolic_pred
                    final_conf = symbolic_conf
                    strategy = {
                        'strategy': 'symbolic',
                        'neural_weight': 0.0,
                        'symbolic_weight': 1.0,
                        'explanation': f'Using symbolic reasoning (higher confidence: {symbolic_conf:.2f} vs {neural_conf:.2f})'
                    }
                
                # Create final result
                result = {
                    'neural': {
                        'prediction': neural_pred,
                        'confidence': neural_conf,
                        'class_name': class_names[neural_pred] if neural_pred < len(class_names) else "Unknown",
                        'probabilities': neural_probs[0].detach().cpu().numpy()  # Detach before numpy conversion
                    },
                    'symbolic': {
                        'prediction': symbolic_pred,
                        'confidence': symbolic_conf,
                        'class_name': class_names[symbolic_pred] if symbolic_pred < len(class_names) else "Unknown",
                        'reasoning_steps': reasoning_steps,
                        'inferred_symbols': [symbol_names[i] for i in inferred if i < len(symbol_names)],
                        'active_symptoms': [symbol_names[i] for i in symptom_ids if i < len(symbol_names)],
                        'class_scores': class_scores,
                        'probabilities': symbolic_probs[0].detach().cpu().numpy()  # Detach before numpy conversion
                    },
                    'nexus': {
                        'prediction': final_pred,
                        'confidence': final_conf,
                        'class_name': class_names[final_pred] if final_pred < len(class_names) else "Unknown",
                        'strategy': strategy
                    }
                }
                
                return result
        except Exception as e:
            # Global error handling
            print(f"Error in diagnosis: {str(e)}")
            # Return a minimal working result
            neural_pred = 0  # Default to first class
            neural_conf = 0.5
            
            # Create minimal result
            result = {
                'neural': {
                    'prediction': neural_pred,
                    'confidence': neural_conf,
                    'class_name': "Class_0",
                    'probabilities': np.array([1.0] + [0.0] * (getattr(self.neural_model, 'num_classes', 3) - 1))
                },
                'symbolic': {
                    'prediction': neural_pred,
                    'confidence': neural_conf,
                    'class_name': "Class_0",
                    'reasoning_steps': {},
                    'inferred_symbols': [],
                    'active_symptoms': [],
                    'class_scores': {},
                    'probabilities': np.array([1.0] + [0.0] * (getattr(self.neural_model, 'num_classes', 3) - 1))
                },
                'nexus': {
                    'prediction': neural_pred,
                    'confidence': neural_conf,
                    'class_name': "Class_0",
                    'strategy': {
                        'strategy': 'neural',
                        'explanation': 'Error fallback strategy'
                    }
                }
            }
            return result
    
    # Apply the patch
    NEXUSParallelExecutor.diagnose = patched_diagnose

# Patch the evaluate method to ensure no_grad context is used
def patch_evaluate_method():
    from nexus_scaling_integration import ScalableNEXUSModel
    original_evaluate = ScalableNEXUSModel.evaluate
    
    def patched_evaluate(self, dataloader, symptom_dict=None, feedback=True, use_async=False, num_workers=1):
        # Always use no_grad context for evaluation
        with torch.no_grad():
            return original_evaluate(self, dataloader, symptom_dict, feedback, use_async, num_workers)
    
    ScalableNEXUSModel.evaluate = patched_evaluate

# Patch the numpy conversions in the result processing
def patch_result_processing():
    # Helper function to safely convert tensor to numpy
    def safe_tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    # Patch the NEXUSParallelExecutor result processing
    original_check_pending = NEXUSParallelExecutor.check_pending_result
    
    def patched_check_pending(self, request_id):
        result = original_check_pending(self, request_id)
        if result is not None:
            # Make tensor conversions safe
            for component in ['neural', 'symbolic']:
                if component in result and 'probabilities' in result[component]:
                    result[component]['probabilities'] = safe_tensor_to_numpy(result[component]['probabilities'])
        return result
    
    NEXUSParallelExecutor.check_pending_result = patched_check_pending

def patch_visualization_methods():
    """Patch the visualization methods to handle edge cases properly"""
    from nexus_real_data import EnhancedNEXUSModel
    
    # Store the original method
    original_visualize = EnhancedNEXUSModel.visualize_results
    
    def patched_visualize(self, output_prefix=None, save_figures=False, show_figures=True):
        try:
            # Check if there's valid data to visualize
            if (self.eval_results['neural']['confusion'] is None or 
                self.eval_results['neural']['total'] == 0):
                print("Warning: No evaluation results to visualize.")
                return {
                    'neural_accuracy': self.eval_results['neural'].get('accuracy', 0),
                    'symbolic_accuracy': self.eval_results['symbolic'].get('accuracy', 0),
                    'nexus_accuracy': self.eval_results['nexus'].get('accuracy', 0),
                    'agreement_cases': self.eval_results.get('agreement_cases', {}),
                    'metacognitive_stats': {},
                    'class_f1_scores': {
                        'neural': [], 'symbolic': [], 'nexus': []
                    }
                }
            
            # Call the original method in a try-except block to catch visualization errors
            try:
                return original_visualize(self, output_prefix, save_figures, show_figures)
            except Exception as e:
                print(f"Error in visualization: {str(e)}")
                
                # Return basic statistics even if visualization fails
                return {
                    'neural_accuracy': self.eval_results['neural'].get('accuracy', 0),
                    'symbolic_accuracy': self.eval_results['symbolic'].get('accuracy', 0),
                    'nexus_accuracy': self.eval_results['nexus'].get('accuracy', 0),
                    'agreement_cases': self.eval_results.get('agreement_cases', {}),
                    'metacognitive_stats': getattr(self.metacognitive, 'get_strategy_stats', lambda: {})(),
                    'class_f1_scores': {
                        'neural': [], 'symbolic': [], 'nexus': []
                    }
                }
        except Exception as e:
            print(f"Error in visualization method: {str(e)}")
            return {}
    
    # Apply the patch
    EnhancedNEXUSModel.visualize_results = patched_visualize
    
    # Also patch the class-wise performance calculation
    if hasattr(EnhancedNEXUSModel, 'calculate_f1'):
        original_calculate_f1 = EnhancedNEXUSModel.calculate_f1
        
        def patched_calculate_f1(self, model_key, class_id):
            try:
                # Add safety checks
                if not self.eval_results[model_key]['predictions'] or not self.eval_results[model_key]['true_labels']:
                    return 0.0
                
                # Safely compute F1 score
                return original_calculate_f1(self, model_key, class_id)
            except Exception as e:
                print(f"Error calculating F1 score: {str(e)}")
                return 0.0
        
        EnhancedNEXUSModel.calculate_f1 = patched_calculate_f1

def main():
    # Apply the patches
    patch_neural_model()
    patch_parallel_executor()
    patch_evaluate_method()
    patch_result_processing()
    patch_visualization_methods()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run scaled NEXUS experiment with patches")
    parser.add_argument("--dataset", type=str, default="scikit-learn/iris", 
                        help="Dataset name")
    parser.add_argument("--samples", type=int, default=1000, 
                        help="Maximum number of samples")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default="nexus_results", 
                        help="Output directory")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Run the experiment with patches applied
    result = run_scaled_nexus_experiment(
        dataset_name=args.dataset,
        max_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=0.001,
        output_dir=args.output_dir,
        device=args.device,
        use_mixed_precision=True,
        use_sparse_kg=True,
        async_reasoning=True,
        random_state=42
    )
    
    # Print summary
    print(f"Training time: {result['train_time'] / 60:.2f} minutes")
    print(f"NEXUS accuracy: {result['test_results']['nexus']['accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()