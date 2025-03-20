"""
Complete Model Replacement approach: Create a custom MedicalTransformerModel class 
that inherits from ScalableTransformerModel and adds medical-specific functionality.

This creates a specialized model class for medical data that properly handles
the requirements for neural-symbolic integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

# Import the base model
from efficient_transformer import ScalableTransformerModel, MixedPrecisionTrainer

class MedicalTransformerModel(ScalableTransformerModel):
    """
    A specialized transformer model for medical data that properly supports
    neural-symbolic integration.
    """
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 feature_names: List[str],
                 class_names: List[str],
                 embed_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 ff_dim: int = 512,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 use_flash_attention: bool = False,
                 use_mixed_precision: bool = False):
        """
        Initialize the medical transformer model.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            feature_names: Names of input features (used as symbols in knowledge graph)
            class_names: Names of output classes
            Other arguments are passed to the base ScalableTransformerModel
        """
        # Call the parent class constructor
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            use_flash_attention=use_flash_attention,
            use_mixed_precision=use_mixed_precision
        )
        
        # Store input_dim - this is the critical fix
        self.input_dim = input_dim
        
        # Add medical-specific attributes
        self.num_symbols = input_dim
        self.symbol_names = feature_names
        self.class_names = class_names
        
        # Medical-specific parameters
        self.risk_factor_weights = nn.Parameter(torch.ones(input_dim) * 0.5)
        self.clinical_thresholds = nn.Parameter(torch.ones(input_dim) * 0.7)
        self.condition_indicators = {}  # Maps medical conditions to their key indicators
        
        # Initialize medical condition indicators
        self._init_medical_conditions()
        
    def _init_medical_conditions(self):
        """Initialize common medical condition indicators based on feature names"""
        # Common indicators for heart disease
        heart_indicators = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'exang', 'oldpeak', 'ca', 'thal'
        ]
        
        # Map features to indices
        feature_to_idx = {name: i for i, name in enumerate(self.symbol_names)}
        
        # Create indicators for heart disease
        self.condition_indicators['heart_disease'] = [
            feature_to_idx[name] for name in heart_indicators 
            if name in feature_to_idx
        ]
        
    def medical_preprocessing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply medical-specific preprocessing to input features.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Preprocessed features
        """
        # Amplify known risk factors based on learned weights
        amplification = torch.sigmoid(self.risk_factor_weights)
        amplified_x = x * (1.0 + 0.2 * amplification)
        
        # Apply clinical thresholds for feature activation
        thresholds = torch.sigmoid(self.clinical_thresholds)
        # Scale thresholds based on the magnitude of the input to avoid over-suppression
        scaled_thresholds = thresholds * torch.mean(x, dim=1, keepdim=True)
        
        # Create activation mask based on thresholds
        mask = (x > scaled_thresholds).float()
        
        # Apply the mask with a residual connection to avoid completely zeroing out values
        processed_x = x * 0.8 + amplified_x * mask * 0.2
        
        return processed_x
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                output_hidden_states: bool = True,
                use_checkpoint: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with medical preprocessing.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            mask: Optional attention mask
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            use_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            logits: Classification logits
            hidden_states: Hidden representation
            attentions: Optional attention weights
        """
        # Apply medical preprocessing if input is 2D (no sequence dimension)
        if x.dim() == 2:
            x = self.medical_preprocessing(x)
            x = x.unsqueeze(1)  # Add sequence dimension
        elif x.dim() == 3:
            # If input already has sequence dimension, apply preprocessing to each step
            batch_size, seq_len, input_dim = x.size()
            x_reshaped = x.view(-1, input_dim)  # Combine batch and sequence dims
            x_processed = self.medical_preprocessing(x_reshaped)
            x = x_processed.view(batch_size, seq_len, input_dim)  # Restore shape
        
        # Call the parent class forward method
        return super().forward(x, mask, output_attentions, output_hidden_states, use_checkpoint)
    
    def extract_medical_features(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        Extract medically relevant features from the hidden states.
        
        Args:
            hidden_states: Hidden representation from the transformer
            
        Returns:
            Dictionary of medical features and their values
        """
        # Get the representation from the first token
        if hidden_states.dim() == 3:
            hidden = hidden_states[:, 0, :]  # [batch_size, hidden_dim]
        else:
            hidden = hidden_states  # Already [batch_size, hidden_dim]
            
        # Extract features for each condition
        features = {}
        
        # Heart disease features
        if 'heart_disease' in self.condition_indicators:
            # Calculate average activation for heart disease indicators
            indicator_indices = self.condition_indicators['heart_disease']
            if indicator_indices:
                # Project hidden state to input space
                if hasattr(self, 'classifier') and hasattr(self.classifier[0], 'weight'):
                    # Use the first classifier layer's weight matrix shape to determine projection
                    input_projection_dim = min(self.input_dim, self.classifier[0].weight.size(1))
                    projection = torch.matmul(hidden, self.classifier[0].weight[:, :input_projection_dim])
                else:
                    # Fallback - just use the hidden state directly
                    projection = hidden
                
                # Safely get activations for heart disease indicators that are within range
                valid_indices = [i for i in indicator_indices if i < projection.size(1)]
                if valid_indices:
                    heart_activations = torch.mean(
                        torch.stack([projection[:, i] for i in valid_indices], dim=1),
                        dim=1
                    )
                    features['heart_disease_activation'] = heart_activations.mean().item()
                else:
                    features['heart_disease_activation'] = 0.0
        
        return features

class MedicalNEXUSModel:
    """
    A specialized NEXUS model for medical data that integrates the MedicalTransformerModel
    with enhanced symbolic reasoning for medical diagnosis.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 num_symbols: int,
                 symbol_names: List[str],
                 class_names: List[str],
                 embed_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 ff_dim: int = 512,
                 device: str = "cpu"):
        """
        Initialize the medical NEXUS model.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            num_symbols: Number of symbols for knowledge graph
            symbol_names: Names of symbols (usually feature names)
            class_names: Names of output classes
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            device: Device to use
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_symbols = num_symbols
        self.symbol_names = symbol_names
        self.class_names = class_names
        self.embed_dim = embed_dim
        self.device = device
        self.symbol_to_id = {name: i for i, name in enumerate(symbol_names)}
        
        # Initialize the neural model as a MedicalTransformerModel
        self.neural_model = MedicalTransformerModel(
            input_dim=input_dim,
            num_classes=num_classes,
            feature_names=symbol_names,
            class_names=class_names,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim
        ).to(device)
        
        # Import necessary components from nexus_real_data
        from nexus_real_data import (
            EnhancedKnowledgeGraph, 
            AdvancedNeuralSymbolicInterface,
            AdvancedMetacognitiveController
        )
        
        # Initialize the knowledge graph
        self.knowledge_graph = EnhancedKnowledgeGraph()
        
        # Initialize the neural-symbolic interface
        self.interface = AdvancedNeuralSymbolicInterface(
            hidden_dim=embed_dim,
            num_symbols=num_symbols,
            num_classes=num_classes
        ).to(device)
        
        # Initialize the metacognitive controller with medical-specific thresholds
        self.metacognitive = AdvancedMetacognitiveController(
            neural_threshold=0.82,  # Higher threshold for neural predictions in medical domain
            symbolic_threshold=0.70  # Slightly lower threshold for symbolic reasoning
        )
        
        # Initialize medical-specific components
        self.risk_levels = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7
        }
        
        # Add clinical guidelines as weights for the metacognitive controller
        self.clinical_guidelines = {
            'symptom_count_threshold': 3,  # Minimum symptoms for high confidence
            'critical_symptoms': ['cp', 'exang', 'ca'],  # Critical symptoms that increase symbolic weight
            'risk_factor_boost': 0.2  # How much to boost confidence for high-risk patients
        }
        
        # Track evaluation results
        self.eval_results = {
            'neural': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'symbolic': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'nexus': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []}
        }
        
        # Case tracking
        self.case_details = []
        
    def init_knowledge_graph(self):
        """Initialize the knowledge graph with medical domain knowledge"""
        kg = self.knowledge_graph
        kg.symbol_offset = self.num_symbols
        kg.num_classes = self.num_classes
        
        # Add entities for features/symptoms
        for i, symbol_name in enumerate(self.symbol_names):
            kg.add_entity(i, symbol_name)
            
        # Add class entities
        for i, class_name in enumerate(self.class_names):
            kg.add_entity(self.num_symbols + i, class_name)
            
        # Add basic relationships
        # This should be extended for specific medical domains
        if len(self.symbol_names) > 0 and self.num_classes > 0:
            # Example relationship
            kg.add_relation(0, "indicates", self.num_symbols, weight=0.6)
            
            # Example rule
            if len(self.symbol_names) > 1:
                kg.add_rule([0, 1], self.num_symbols + 1, confidence=0.7)
                
        return kg
    
    def forward(self, x):
        """Forward pass through the neural model"""
        x = x.to(self.device)
        return self.neural_model(x)[0]
    
    def train_neural(self, dataloader, num_epochs=5, lr=0.001, scheduler=None, weight_decay=1e-5):
        """Train the neural component with medical-specific optimizations"""
        self.neural_model.train()
        
        # Use AdamW optimizer with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.neural_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Set up scheduler
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
            
        # Use a weighted loss function for medical data to handle class imbalance
        class_weights = torch.ones(self.num_classes, device=self.device)
        if self.num_classes == 2:  # Binary classification - give higher weight to positive class
            class_weights[1] = 1.5
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        epoch_stats = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Track progress
            import tqdm
            progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _, _ = self.neural_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update scheduler if using cosine
                if scheduler == 'cosine' and scheduler_obj is not None:
                    scheduler_obj.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item() * labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
            
            # Calculate epoch statistics
            avg_loss = epoch_loss / total
            accuracy = 100 * correct / total
            
            # Update reduce-on-plateau scheduler if used
            if scheduler == 'reduce' and scheduler_obj is not None:
                scheduler_obj.step(avg_loss)
                
            # Print epoch results
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Store epoch stats
            epoch_stats.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
        return epoch_stats
    
    def diagnose(self, x, active_symptoms=None, risk_level='medium'):
        """
        Diagnose a patient using neural-symbolic integration with medical enhancements.
        
        Args:
            x: Input features
            active_symptoms: Optional list of active symptoms
            risk_level: Patient risk level ('low', 'medium', 'high')
            
        Returns:
            Diagnosis result with neural, symbolic, and integrated predictions
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        with torch.no_grad():
            # Get neural prediction
            neural_logits, neural_repr, _ = self.neural_model(x)
            neural_probs = F.softmax(neural_logits, dim=1)
            neural_pred = torch.argmax(neural_probs, dim=1).item()
            neural_conf = neural_probs[0, neural_pred].item()
            
            # Get neural-to-symbolic mapping
            symbolic_activations, similarities, _ = self.interface.neural_to_symbolic(neural_repr)
            
            # Get active symptoms
            if active_symptoms is not None:
                # Use provided symptoms
                symptom_ids = [self.symbol_to_id[name] for name in active_symptoms 
                             if name in self.symbol_to_id]
            else:
                # Extract from neural representation
                symptom_ids = torch.nonzero(symbolic_activations[0]).squeeze(-1).tolist()
                if not isinstance(symptom_ids, list):
                    symptom_ids = [symptom_ids]
            
            # Apply medical domain knowledge for reasoning
            inferred, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids)
            
            # Process symbolic results
            symbolic_scores = torch.zeros(1, self.num_classes, device=self.device)
            for class_id, score in class_scores.items():
                if class_id < self.num_classes:
                    symbolic_scores[0, class_id] = score
                    
            if symbolic_scores.sum() == 0:
                symbolic_probs = torch.ones(1, self.num_classes, device=self.device) / self.num_classes
            else:
                symbolic_probs = F.softmax(symbolic_scores, dim=1)
                
            symbolic_pred = torch.argmax(symbolic_probs, dim=1).item()
            symbolic_conf = symbolic_probs[0, symbolic_pred].item()
            
            # Medical-specific adjustments to confidence
            # Adjust neural confidence based on risk level
            risk_factor = self.risk_levels.get(risk_level, 0.5)
            if risk_level == 'high':
                # For high-risk patients, reduce neural confidence slightly 
                # to favor symbolic reasoning based on clinical guidelines
                neural_conf *= 0.9
            
            # Adjust symbolic confidence based on critical symptoms
            critical_symptom_count = len(set(symptom_ids).intersection(
                set(self.symbol_to_id[name] for name in self.clinical_guidelines['critical_symptoms']
                    if name in self.symbol_to_id)))
            
            if critical_symptom_count > 0:
                # Boost symbolic confidence if critical symptoms are present
                symbolic_conf = min(1.0, symbolic_conf * (1.0 + critical_symptom_count * 0.1))
                
            # Make final decision with metacognitive controller
            strategy = self.metacognitive.decide_strategy(neural_conf, symbolic_conf, risk_level)
            
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
            
            # Extract medical features - with try/except for safety
            try:
                medical_features = self.neural_model.extract_medical_features(neural_repr)
            except Exception as e:
                # Just create some basic medical features as fallback
                medical_features = {
                    'heart_disease_activation': neural_conf * 0.8 if neural_pred == 1 else 0.2
                }
        
        # Create result dictionary
        result = {
            'neural': {
                'prediction': neural_pred,
                'confidence': neural_conf,
                'class_name': self.class_names[neural_pred] if neural_pred < len(self.class_names) else "Unknown",
                'probabilities': neural_probs[0].cpu().numpy()
            },
            'symbolic': {
                'prediction': symbolic_pred,
                'confidence': symbolic_conf,
                'class_name': self.class_names[symbolic_pred] if symbolic_pred < len(self.class_names) else "Unknown",
                'reasoning_steps': reasoning_steps,
                'inferred_symbols': [self.symbol_names[i] for i in inferred if i < len(self.symbol_names)],
                'active_symptoms': [self.symbol_names[i] for i in symptom_ids if i < len(self.symbol_names)],
                'class_scores': class_scores,
                'probabilities': symbolic_probs[0].cpu().numpy()
            },
            'nexus': {
                'prediction': final_pred,
                'confidence': final_conf,
                'class_name': self.class_names[final_pred] if final_pred < len(self.class_names) else "Unknown",
                'strategy': strategy
            }
        }
        
        # Add medical-specific information
        result['medical'] = {
            'risk_level': risk_level,
            'critical_symptoms': [s for s in self.symbol_names if s in self.clinical_guidelines['critical_symptoms'] 
                                 and self.symbol_to_id[s] in symptom_ids],
            'medical_features': medical_features
        }
        
        return result
    
    def evaluate(self, dataloader, symptom_dict=None, feedback=True, use_async=False):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for the test set
            symptom_dict: Optional dictionary mapping sample indices to active symptoms
            feedback: Whether to provide feedback to the metacognitive controller
            use_async: Whether to use asynchronous processing
            
        Returns:
            Evaluation results
        """
        self.neural_model.eval()
        self.interface.eval()
        
        # Initialize results
        for key in self.eval_results:
            self.eval_results[key]['correct'] = 0
            self.eval_results[key]['total'] = 0
            self.eval_results[key]['predictions'] = []
            self.eval_results[key]['true_labels'] = []
            self.eval_results[key]['confidence'] = []
            
        # Create confusion matrices
        self.eval_results['neural']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['symbolic']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['nexus']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        
        # Track agreement
        agreement_cases = {
            'all_correct': 0, 
            'all_wrong': 0, 
            'neural_only': 0, 
            'symbolic_only': 0, 
            'nexus_better': 0
        }
        
        # Clear case details
        self.case_details = []
        
        # Evaluation loop
        sample_index = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                batch_size = inputs.size(0)
                
                for j in range(batch_size):
                    # Get active symptoms for this sample if available
                    active_symptoms = symptom_dict.get(sample_index, None) if symptom_dict else None
                    
                    # Get diagnosis
                    sample_input = inputs[j].unsqueeze(0)
                    true_label = labels[j].item()
                    
                    try:
                        # Get diagnosis
                        result = self.diagnose(sample_input, active_symptoms)
                        
                        # Create case detail
                        case_detail = {
                            'index': sample_index,
                            'true_label': true_label,
                            'true_class': self.class_names[true_label],
                            'neural_pred': result['neural']['prediction'],
                            'neural_conf': result['neural']['confidence'],
                            'symbolic_pred': result['symbolic']['prediction'],
                            'symbolic_conf': result['symbolic']['confidence'],
                            'nexus_pred': result['nexus']['prediction'],
                            'nexus_conf': result['nexus']['confidence'],
                            'nexus_strategy': result['nexus']['strategy']['strategy'],
                            'active_symptoms': active_symptoms,
                            'critical_symptoms': result['medical']['critical_symptoms'] if 'medical' in result else [],
                            'risk_level': result['medical']['risk_level'] if 'medical' in result else 'medium'
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
                            
                        # Track agreement
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
                            
                        # Update metacognitive controller
                        if feedback:
                            self.metacognitive.update_thresholds(
                                neural_correct, 
                                symbolic_correct,
                                result['nexus']['strategy']['strategy']
                            )
                    except Exception as e:
                        print(f"Error processing sample {sample_index}: {str(e)}")
                        
                    sample_index += 1
                    
        # Calculate final metrics
        for key in self.eval_results:
            if self.eval_results[key]['total'] > 0:
                self.eval_results[key]['accuracy'] = (
                    self.eval_results[key]['correct'] / self.eval_results[key]['total']
                )
            else:
                self.eval_results[key]['accuracy'] = 0
                
        self.eval_results['agreement_cases'] = agreement_cases
        
        return self.eval_results
        
    def explain_diagnosis(self, result, detail_level='medium', include_confidence=True):
        """
        Generate a human-readable explanation for a diagnosis result.
        
        Args:
            result: Diagnosis result from diagnose method
            detail_level: Level of detail ('simple', 'medium', 'high')
            include_confidence: Whether to include confidence scores
            
        Returns:
            String explanation of the diagnosis
        """
        from nexus_real_data import EnhancedNEXUSModel
        
        # Use the original explanation generator for basic explanation
        dummy_model = EnhancedNEXUSModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            num_symbols=self.num_symbols,
            symbol_names=self.symbol_names,
            class_names=self.class_names
        )
        
        # Get base explanation
        base_explanation = dummy_model.explain_diagnosis(result, detail_level, include_confidence)
        
        # Add medical-specific explanation if available
        if 'medical' in result:
            med_explanation = ["\nMedical Analysis:"]
            
            # Add risk level
            med_explanation.append(f"Patient Risk Level: {result['medical']['risk_level'].capitalize()}")
            
            # Add critical symptoms if present
            if result['medical']['critical_symptoms']:
                med_explanation.append(f"Critical Symptoms: {', '.join(result['medical']['critical_symptoms'])}")
                
            # Add medical features if available
            if 'medical_features' in result['medical']:
                med_explanation.append("Medical Indicators:")
                for feature, value in result['medical']['medical_features'].items():
                    med_explanation.append(f"  - {feature.replace('_', ' ').title()}: {value:.2f}")
                    
            # Add clinical guidelines if high detail level
            if detail_level == 'high':
                med_explanation.append("\nClinical Guidelines Applied:")
                med_explanation.append(f"  - Critical symptom threshold: {len(self.clinical_guidelines['critical_symptoms'])}")
                med_explanation.append(f"  - Risk factor influence: {self.risk_levels[result['medical']['risk_level']]:.2f}")
                
            # Add the medical explanation to the base explanation
            return base_explanation + "\n" + "\n".join(med_explanation)
        
        return base_explanation
    
    def export_results(self, filename):
        """
        Export evaluation results to a CSV file.
        
        Args:
            filename: Output file path
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            import pandas as pd
            
            if not self.case_details:
                print("No case details available. Run evaluate() first.")
                return False
                
            df = pd.DataFrame(self.case_details)
            
            # Add correctness columns
            df['neural_correct'] = df['neural_pred'] == df['true_label']
            df['symbolic_correct'] = df['symbolic_pred'] == df['true_label']
            df['nexus_correct'] = df['nexus_pred'] == df['true_label']
            df['nexus_improved'] = ((~df['neural_correct'] | ~df['symbolic_correct']) & df['nexus_correct'])
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting results: {str(e)}")
            return False
    
    def visualize_results(self, output_prefix=None, save_figures=False, show_figures=True):
        """
        Visualize evaluation results with medical-specific metrics.
        
        Args:
            output_prefix: Prefix for output files
            save_figures: Whether to save figures to disk
            show_figures: Whether to display figures
            
        Returns:
            Summary dictionary
        """
        # Use the visualization from the original model
        from nexus_real_data import EnhancedNEXUSModel
        
        # Create a temporary model to use its visualization method
        temp_model = EnhancedNEXUSModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            num_symbols=self.num_symbols,
            symbol_names=self.symbol_names,
            class_names=self.class_names
        )
        
        # Copy evaluation results
        temp_model.eval_results = self.eval_results
        temp_model.case_details = self.case_details
        
        # Call the visualization method
        summary = temp_model.visualize_results(output_prefix, save_figures, show_figures)
        
        # Add medical-specific visualizations
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        
        try:
            if self.case_details:
                # Create a DataFrame from case details
                df = pd.DataFrame(self.case_details)
                
                # Add correctness columns if not present
                if 'neural_correct' not in df.columns:
                    df['neural_correct'] = df['neural_pred'] == df['true_label']
                if 'symbolic_correct' not in df.columns:
                    df['symbolic_correct'] = df['symbolic_pred'] == df['true_label']
                if 'nexus_correct' not in df.columns:
                    df['nexus_correct'] = df['nexus_pred'] == df['true_label']
                
                # 1. Risk Level Analysis
                if 'risk_level' in df.columns:
                    plt.figure(figsize=(14, 6))
                    
                    # Count samples by risk level
                    risk_counts = df['risk_level'].value_counts()
                    
                    # Accuracy by risk level for each component
                    risk_level_data = []
                    for risk in df['risk_level'].unique():
                        risk_df = df[df['risk_level'] == risk]
                        risk_level_data.append({
                            'Risk Level': risk,
                            'Neural Accuracy': risk_df['neural_correct'].mean() * 100,
                            'Symbolic Accuracy': risk_df['symbolic_correct'].mean() * 100,
                            'NEXUS Accuracy': risk_df['nexus_correct'].mean() * 100
                        })
                    
                    risk_df = pd.DataFrame(risk_level_data)
                    
                    # Create the plot
                    ax = sns.barplot(x='Risk Level', y='value', hue='variable', 
                                    data=pd.melt(risk_df, ['Risk Level'], 
                                                ['Neural Accuracy', 'Symbolic Accuracy', 'NEXUS Accuracy']))
                    plt.title('Accuracy by Patient Risk Level')
                    plt.ylabel('Accuracy (%)')
                    plt.ylim(0, 100)
                    
                    # Add counts as text above bars
                    for risk in risk_df['Risk Level']:
                        count = risk_counts[risk]
                        plt.text(list(risk_df['Risk Level']).index(risk), 95, 
                                f"n={count}", ha='center', fontsize=10)
                    
                    plt.tight_layout()
                    if save_figures and output_prefix:
                        plt.savefig(f"{output_prefix}_risk_levels.png", dpi=300, bbox_inches='tight')
                    if show_figures:
                        plt.show()
                    else:
                        plt.close()
                
                # 2. Critical Symptoms Analysis
                if 'critical_symptoms' in df.columns and any(df['critical_symptoms'].map(len) > 0):
                    plt.figure(figsize=(14, 6))
                    
                    # Create binary column for presence of critical symptoms
                    df['has_critical'] = df['critical_symptoms'].map(len) > 0
                    
                    # Group by critical symptom presence
                    critical_data = []
                    for has_critical in [False, True]:
                        critical_df = df[df['has_critical'] == has_critical]
                        if len(critical_df) > 0:
                            critical_data.append({
                                'Critical Symptoms': 'Present' if has_critical else 'Absent',
                                'Neural Accuracy': critical_df['neural_correct'].mean() * 100,
                                'Symbolic Accuracy': critical_df['symbolic_correct'].mean() * 100,
                                'NEXUS Accuracy': critical_df['nexus_correct'].mean() * 100,
                                'Count': len(critical_df)
                            })
                    
                    if critical_data:
                        critical_df = pd.DataFrame(critical_data)
                        
                        # Create the plot
                        ax = sns.barplot(x='Critical Symptoms', y='value', hue='variable', 
                                        data=pd.melt(critical_df, ['Critical Symptoms', 'Count'], 
                                                    ['Neural Accuracy', 'Symbolic Accuracy', 'NEXUS Accuracy']))
                        plt.title('Impact of Critical Symptoms on Accuracy')
                        plt.ylabel('Accuracy (%)')
                        plt.ylim(0, 100)
                        
                        # Add counts as text above bars
                        for i, row in critical_df.iterrows():
                            plt.text(i, 95, f"n={row['Count']}", ha='center', fontsize=10)
                        
                        plt.tight_layout()
                        if save_figures and output_prefix:
                            plt.savefig(f"{output_prefix}_critical_symptoms.png", dpi=300, bbox_inches='tight')
                        if show_figures:
                            plt.show()
                        else:
                            plt.close()
                            
                # 3. Strategy by Risk Level
                if 'risk_level' in df.columns and 'nexus_strategy' in df.columns:
                    plt.figure(figsize=(14, 6))
                    
                    # Crosstab of strategy vs risk level
                    strategy_risk = pd.crosstab(df['nexus_strategy'], df['risk_level'], normalize='columns') * 100
                    strategy_risk.plot(kind='bar', stacked=False)
                    
                    plt.title('Strategy Selection by Risk Level')
                    plt.xlabel('Strategy')
                    plt.ylabel('Percentage (%)')
                    plt.legend(title='Risk Level')
                    
                    plt.tight_layout()
                    if save_figures and output_prefix:
                        plt.savefig(f"{output_prefix}_strategy_by_risk.png", dpi=300, bbox_inches='tight')
                    if show_figures:
                        plt.show()
                    else:
                        plt.close()
                        
                # Add medical metrics to summary
                medical_summary = {}
                
                # Risk level performance
                if 'risk_level' in df.columns:
                    risk_performance = {}
                    for risk in df['risk_level'].unique():
                        risk_df = df[df['risk_level'] == risk]
                        risk_performance[risk] = {
                            'neural_accuracy': risk_df['neural_correct'].mean() * 100,
                            'symbolic_accuracy': risk_df['symbolic_correct'].mean() * 100,
                            'nexus_accuracy': risk_df['nexus_correct'].mean() * 100,
                            'count': len(risk_df)
                        }
                    medical_summary['risk_performance'] = risk_performance
                    
                # Critical symptoms impact
                if 'critical_symptoms' in df.columns:
                    df['has_critical'] = df['critical_symptoms'].map(len) > 0
                    critical_df = df[df['has_critical']]
                    non_critical_df = df[~df['has_critical']]
                    
                    medical_summary['critical_symptoms_impact'] = {
                        'with_critical': {
                            'neural_accuracy': critical_df['neural_correct'].mean() * 100 if len(critical_df) > 0 else 0,
                            'symbolic_accuracy': critical_df['symbolic_correct'].mean() * 100 if len(critical_df) > 0 else 0,
                            'nexus_accuracy': critical_df['nexus_correct'].mean() * 100 if len(critical_df) > 0 else 0,
                            'count': len(critical_df)
                        },
                        'without_critical': {
                            'neural_accuracy': non_critical_df['neural_correct'].mean() * 100 if len(non_critical_df) > 0 else 0,
                            'symbolic_accuracy': non_critical_df['symbolic_correct'].mean() * 100 if len(non_critical_df) > 0 else 0,
                            'nexus_accuracy': non_critical_df['nexus_correct'].mean() * 100 if len(non_critical_df) > 0 else 0,
                            'count': len(non_critical_df)
                        }
                    }
                    
                # Strategy usage by risk level
                if 'risk_level' in df.columns and 'nexus_strategy' in df.columns:
                    strategy_counts = {}
                    for risk in df['risk_level'].unique():
                        risk_df = df[df['risk_level'] == risk]
                        strategies = risk_df['nexus_strategy'].value_counts(normalize=True) * 100
                        strategy_counts[risk] = {strategy: pct for strategy, pct in strategies.items()}
                        
                    medical_summary['strategy_by_risk'] = strategy_counts
                
                # Add medical summary to the overall summary
                summary['medical'] = medical_summary
        except Exception as e:
            print(f"Error in medical visualizations: {str(e)}")
            
        return summary
        
    def save_model(self, directory):
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            True if successful, False otherwise
        """
        import os
        
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            # Save model configuration
            config = {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'num_symbols': self.num_symbols,
                'embed_dim': self.embed_dim,
                'clinical_guidelines': self.clinical_guidelines,
                'risk_levels': self.risk_levels
            }
            
            import json
            with open(os.path.join(directory, 'config.json'), 'w') as f:
                json.dump(config, f)
                
            # Save symbol and class names
            with open(os.path.join(directory, 'names.json'), 'w') as f:
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
            if not os.path.exists(kg_dir):
                os.makedirs(kg_dir)
                
            if hasattr(self.knowledge_graph, 'save_to_disk'):
                self.knowledge_graph.save_to_disk(kg_dir)
                
            print(f"Model saved to {directory}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    @classmethod
    def load_model(cls, directory, device="cpu"):
        """
        Load a model from disk.
        
        Args:
            directory: Directory containing the saved model
            device: Device to load the model on
            
        Returns:
            Loaded MedicalNEXUSModel
        """
        import os
        
        try:
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
                device=device
            )
            
            # Add clinical guidelines and risk levels
            if 'clinical_guidelines' in config:
                model.clinical_guidelines = config['clinical_guidelines']
                
            if 'risk_levels' in config:
                model.risk_levels = config['risk_levels']
            
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
            if os.path.exists(kg_dir) and hasattr(model.knowledge_graph, 'load_from_disk'):
                model.knowledge_graph.load_from_disk(kg_dir)
                
            print(f"Model loaded from {directory}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

def initialize_heart_disease_knowledge_graph(model):
    """
    Initialize the knowledge graph with domain-specific knowledge for heart disease.
    
    Args:
        model: MedicalNEXUSModel instance
        
    Returns:
        Initialized knowledge graph
    """
    kg = model.knowledge_graph
    kg.symbol_offset = model.num_symbols
    kg.num_classes = model.num_classes
    
    feature_names = model.symbol_names
    
    # Add entities for features
    for i, feature_name in enumerate(feature_names):
        kg.add_entity(i, feature_name)
    
    # Add class entities
    kg.add_entity(kg.symbol_offset, "No Heart Disease")
    kg.add_entity(kg.symbol_offset + 1, "Heart Disease")
    
    # Add medical domain knowledge
    
    # 1. Age factor
    kg.add_entity(100, "Age > 50", {"risk_factor": 0.3, "increases_1": 0.4})
    kg.add_relation(feature_names.index('age'), "indicates", 100, weight=0.7)  # age -> Age > 50
    kg.add_relation(100, "increases_risk", kg.symbol_offset + 1, weight=0.6)  # Age > 50 -> Heart Disease
    
    # 2. Gender factor
    kg.add_entity(101, "Male Gender", {"risk_factor": 0.25, "increases_1": 0.3})
    kg.add_relation(feature_names.index('sex'), "indicates", 101, weight=0.9)  # sex -> Male Gender
    kg.add_relation(101, "increases_risk", kg.symbol_offset + 1, weight=0.5)  # Male Gender -> Heart Disease
    
    # 3. Chest pain types
    kg.add_entity(102, "Typical Angina")
    kg.add_entity(103, "Atypical Angina")
    kg.add_entity(104, "Non-anginal Pain")
    kg.add_entity(105, "Asymptomatic", {"risk_factor": 0.4, "increases_1": 0.6})
    
    kg.add_relation(feature_names.index('cp'), "indicates", 102, weight=0.8)  # cp -> Typical Angina (when cp=1)
    kg.add_relation(feature_names.index('cp'), "indicates", 103, weight=0.8)  # cp -> Atypical Angina (when cp=2)
    kg.add_relation(feature_names.index('cp'), "indicates", 104, weight=0.8)  # cp -> Non-anginal Pain (when cp=3)
    kg.add_relation(feature_names.index('cp'), "indicates", 105, weight=0.8)  # cp -> Asymptomatic (when cp=4)
    
    kg.add_relation(105, "increases_risk", kg.symbol_offset + 1, weight=0.7)  # Asymptomatic -> Heart Disease
    
    # 4. High blood pressure
    kg.add_entity(106, "High Blood Pressure", {"risk_factor": 0.3, "increases_1": 0.4})
    kg.add_relation(feature_names.index('trestbps'), "indicates", 106, weight=0.6)  # trestbps -> High Blood Pressure
    kg.add_relation(106, "increases_risk", kg.symbol_offset + 1, weight=0.5)  # High BP -> Heart Disease
    
    # 5. Cholesterol
    kg.add_entity(107, "High Cholesterol", {"risk_factor": 0.35, "increases_1": 0.4})
    kg.add_relation(feature_names.index('chol'), "indicates", 107, weight=0.6)  # chol -> High Cholesterol
    kg.add_relation(107, "increases_risk", kg.symbol_offset + 1, weight=0.5)  # High Cholesterol -> Heart Disease
    
    # 6. Diabetes indicator
    kg.add_entity(108, "Diabetes", {"risk_factor": 0.4, "increases_1": 0.5})
    kg.add_relation(feature_names.index('fbs'), "indicates", 108, weight=0.7)  # fbs -> Diabetes
    kg.add_relation(108, "increases_risk", kg.symbol_offset + 1, weight=0.6)  # Diabetes -> Heart Disease
    
    # 7. ECG abnormalities
    kg.add_entity(109, "ECG Abnormality", {"risk_factor": 0.3, "increases_1": 0.4})
    kg.add_relation(feature_names.index('restecg'), "indicates", 109, weight=0.6)  # restecg -> ECG Abnormality
    kg.add_relation(109, "increases_risk", kg.symbol_offset + 1, weight=0.5)  # ECG Abnormality -> Heart Disease
    
    # 8. Max heart rate
    kg.add_entity(110, "Low Max Heart Rate", {"risk_factor": 0.25, "increases_1": 0.3})
    kg.add_relation(feature_names.index('thalach'), "indicates", 110, weight=0.5)  # thalach -> Low Max Heart Rate
    kg.add_relation(110, "increases_risk", kg.symbol_offset + 1, weight=0.4)  # Low Max Heart Rate -> Heart Disease
    
    # 9. Exercise induced angina
    kg.add_entity(111, "Exercise Angina", {"risk_factor": 0.4, "increases_1": 0.6})
    kg.add_relation(feature_names.index('exang'), "indicates", 111, weight=0.8)  # exang -> Exercise Angina
    kg.add_relation(111, "increases_risk", kg.symbol_offset + 1, weight=0.7)  # Exercise Angina -> Heart Disease
    
    # 10. ST depression
    kg.add_entity(112, "ST Depression", {"risk_factor": 0.35, "increases_1": 0.5})
    kg.add_relation(feature_names.index('oldpeak'), "indicates", 112, weight=0.7)  # oldpeak -> ST Depression
    kg.add_relation(112, "increases_risk", kg.symbol_offset + 1, weight=0.6)  # ST Depression -> Heart Disease
    
    # 11. Slope of ST segment
    kg.add_entity(113, "Abnormal ST Slope", {"risk_factor": 0.3, "increases_1": 0.4})
    kg.add_relation(feature_names.index('slope'), "indicates", 113, weight=0.6)  # slope -> Abnormal ST Slope
    kg.add_relation(113, "increases_risk", kg.symbol_offset + 1, weight=0.5)  # Abnormal ST Slope -> Heart Disease
    
    # 12. Number of vessels colored by fluoroscopy
    kg.add_entity(114, "Multiple Vessels", {"risk_factor": 0.5, "increases_1": 0.7})
    kg.add_relation(feature_names.index('ca'), "indicates", 114, weight=0.8)  # ca -> Multiple Vessels
    kg.add_relation(114, "increases_risk", kg.symbol_offset + 1, weight=0.8)  # Multiple Vessels -> Heart Disease
    
    # 13. Thalassemia
    kg.add_entity(115, "Reversible Defect", {"risk_factor": 0.45, "increases_1": 0.6})
    kg.add_relation(feature_names.index('thal'), "indicates", 115, weight=0.7)  # thal -> Reversible Defect
    kg.add_relation(115, "increases_risk", kg.symbol_offset + 1, weight=0.7)  # Reversible Defect -> Heart Disease
    
    # Add rules (combinations of factors)
    
    # Rule 1: Age + Gender + Chest Pain
    kg.add_rule([100, 101, 105], kg.symbol_offset + 1, confidence=0.85)  # Older men with asymptomatic chest pain
    
    # Rule 2: Diabetes + High BP + High Cholesterol
    kg.add_rule([106, 107, 108], kg.symbol_offset + 1, confidence=0.8)  # High BP, high cholesterol, and diabetes
    
    # Rule 3: Exercise Angina + ST depression
    kg.add_rule([111, 112], kg.symbol_offset + 1, confidence=0.8)  # Exercise angina and ST depression
    
    # Rule 4: Multiple Vessels + Reversible Defect
    kg.add_rule([114, 115], kg.symbol_offset + 1, confidence=0.9)  # Multiple vessels with reversible defect
    
    # Rule 5: Age + Diabetes + Exercise Angina
    kg.add_rule([100, 108, 111], kg.symbol_offset + 1, confidence=0.85)  # Older people with diabetes and exercise angina
    
    # Hierarchy relationships
    kg.add_hierarchy(102, "Chest Pain")  # Typical Angina is a type of Chest Pain
    kg.add_hierarchy(103, "Chest Pain")  # Atypical Angina is a type of Chest Pain
    kg.add_hierarchy(104, "Chest Pain")  # Non-anginal Pain is a type of Chest Pain
    kg.add_hierarchy(105, "Chest Pain")  # Asymptomatic is a type of Chest Pain
    
    # Common risk factors group
    kg.add_hierarchy(106, "Cardiovascular Risk Factor")  # High BP is a Cardiovascular Risk Factor
    kg.add_hierarchy(107, "Cardiovascular Risk Factor")  # High Cholesterol is a Cardiovascular Risk Factor
    kg.add_hierarchy(108, "Cardiovascular Risk Factor")  # Diabetes is a Cardiovascular Risk Factor
    
    return kg

# Example usage
if __name__ == "__main__":
    import torch
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import Dataset, DataLoader
    
    print("Medical Transformer Model Test")
    print("==============================")
    
    # Simple dataset for testing
    class HeartDiseaseDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            
        def __len__(self):
            return len(self.features)
            
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    # Load UCI Heart Disease dataset
    print("Loading UCI Heart Disease dataset...")
    
    # Example feature names for the Heart Disease dataset
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Create synthetic test data (replace with actual data loading)
    import numpy as np
    X = np.random.rand(100, len(feature_names))
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create DataLoaders
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    test_dataset = HeartDiseaseDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create and initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MedicalNEXUSModel(
        input_dim=len(feature_names),
        num_classes=2,
        num_symbols=len(feature_names),
        symbol_names=feature_names,
        class_names=["No Heart Disease", "Heart Disease"],
        embed_dim=64,
        device=device
    )
    
    # Initialize knowledge graph
    print("Initializing knowledge graph...")
    initialize_heart_disease_knowledge_graph(model)
    
    # Train the model
    print("Training model...")
    model.train_neural(train_loader, num_epochs=2, lr=0.001)
    
    # Evaluate the model
    print("Evaluating model...")
    results = model.evaluate(test_loader)
    
    # Print results
    print(f"Neural accuracy: {results['neural']['accuracy']*100:.2f}%")
    print(f"Symbolic accuracy: {results['symbolic']['accuracy']*100:.2f}%")
    print(f"NEXUS accuracy: {results['nexus']['accuracy']*100:.2f}%")
    
    # Test diagnosis
    print("\nGenerating example diagnosis...")
    sample_input = X_test[0:1]
    diagnosis = model.diagnose(torch.tensor(sample_input, dtype=torch.float32))
    
    print("Diagnosis class:", diagnosis['nexus']['class_name'])
    print("Strategy used:", diagnosis['nexus']['strategy']['strategy'])
    print("Confidence:", diagnosis['nexus']['confidence'])
    
    # Generate explanation
    explanation = model.explain_diagnosis(diagnosis)
    print("\nDiagnosis Explanation:")
    print(explanation)