import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from tabulate import tabulate
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict
import os
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ===========================
# 1. Enhanced Neural Component
# ===========================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        output, attn_weights = self.attention(q, k, v, mask)
        
        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(output)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        residual = x
        x = self.norm1(x)
        x_attn, attn_weights = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x_attn)
        
        # Feed forward
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        
        return x, attn_weights

class AdvancedNeuralModel(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_layers=3, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # Convert to batch_size x 1 x input_dim and embed if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.embedding(x)
        
        # Pass through transformer layers
        attentions = []
        for layer in self.transformer_layers:
            x, attn = layer(x)
            attentions.append(attn)
            
        # Use the representation of the first token for classification
        x = x.squeeze(1) if x.size(1) == 1 else x[:, 0]
        
        # Classify
        logits = self.classifier(x)
        
        return logits, x, attentions

# ===========================
# 2. Enhanced Symbolic Component with Knowledge Graph
# ===========================
class EnhancedKnowledgeGraph:
    def __init__(self):
        self.entities = {}                 # entity_id -> name
        self.relations = []                # (source_id, relation_type, target_id, weight)
        self.rules = []                    # (premise_ids, conclusion_id, confidence)
        self.hierarchy = defaultdict(set)  # entity_id -> set of parent entity_ids
        self.entity_attrs = {}             # entity_id -> {attribute: value}
        self.symbol_offset = 0  # should be set externally (e.g., number of symbols)
        self.num_classes = 0    # should be set externally
        
    def add_entity(self, entity_id, name, attributes=None):
        """Add an entity to the knowledge graph with optional attributes"""
        self.entities[entity_id] = name
        if attributes:
            self.entity_attrs[entity_id] = attributes
        return self
        
    def add_relation(self, source_id, relation_type, target_id, weight=1.0):
        """Add a relation between two entities with a weight"""
        assert isinstance(source_id, int) and isinstance(target_id, int), "IDs must be integers"
        self.relations.append((source_id, relation_type, target_id, weight))
        return self
        
    def add_rule(self, premise_ids, conclusion_id, confidence=1.0):
        """Add a logical rule with a confidence score"""
        assert all(isinstance(p_id, int) for p_id in premise_ids) and isinstance(conclusion_id, int), "IDs must be integers"
        self.rules.append((premise_ids, conclusion_id, confidence))
        return self
        
    def add_hierarchy(self, child_id, parent_id):
        """Add hierarchical relationship (e.g., specific_fever is a fever)"""
        self.hierarchy[child_id].add(parent_id)
        return self
    
    def get_ancestors(self, entity_id):
        """Get all ancestors of an entity in the hierarchy"""
        ancestors = set()
        to_process = list(self.hierarchy[entity_id])
        
        while to_process:
            parent = to_process.pop()
            ancestors.add(parent)
            to_process.extend(self.hierarchy[parent] - ancestors)
            
        return ancestors
        
    def reason(self, active_entities, max_hops=3):
        """
        Apply enhanced reasoning to derive new knowledge
        """
        # Initialize with active entities and their hierarchical parents
        inferred = set(active_entities)
        for entity in list(active_entities):
            inferred.update(self.get_ancestors(entity))
            
        reasoning_steps = {}
        confidences = {}
        
        # Default class scores (keys 0..num_classes-1)
        class_scores = defaultdict(float)
        
        # Initialize reasoning steps and confidences for active entities
        for entity_id in active_entities:
            if entity_id in self.entities:
                reasoning_steps[entity_id] = f"Given: {self.entities[entity_id]}"
                confidences[entity_id] = 1.0
        
        # Add reasoning steps for ancestor entities
        for entity_id in inferred - set(active_entities):
            if entity_id in self.entities:
                for child in active_entities:
                    if entity_id in self.get_ancestors(child):
                        reasoning_steps[entity_id] = f"Hierarchical: {self.entities[child]} is a type of {self.entities[entity_id]}"
                        confidences[entity_id] = 0.95
                        break
        
        # Multi-hop reasoning
        for _ in range(max_hops):
            new_inferences = set()
            
            # Apply relations
            for source_id, relation_type, target_id, weight in self.relations:
                if source_id in inferred and target_id not in inferred:
                    new_inferences.add(target_id)
                    step = f"{self.entities[source_id]} --{relation_type}--> {self.entities[target_id]}"
                    reasoning_steps[target_id] = step
                    confidences[target_id] = weight * confidences.get(source_id, 1.0)
                    
                    if target_id >= self.symbol_offset and target_id < self.symbol_offset + self.num_classes:
                        key = target_id - self.symbol_offset
                        class_scores[key] = max(class_scores[key], confidences[target_id])
            
            # Apply rules
            for premise_ids, conclusion_id, confidence in self.rules:
                if all(p_id in inferred for p_id in premise_ids) and conclusion_id not in inferred:
                    new_inferences.add(conclusion_id)
                    premises = [self.entities[p_id] for p_id in premise_ids]
                    step = f"Rule: IF {' AND '.join(premises)} THEN {self.entities[conclusion_id]}"
                    reasoning_steps[conclusion_id] = step
                    
                    premise_conf = min([confidences.get(p_id, 1.0) for p_id in premise_ids])
                    rule_conf = confidence * premise_conf
                    confidences[conclusion_id] = rule_conf
                    
                    if conclusion_id >= self.symbol_offset and conclusion_id < self.symbol_offset + self.num_classes:
                        key = conclusion_id - self.symbol_offset
                        class_scores[key] = max(class_scores[key], rule_conf)
            
            if not new_inferences:
                break
                
            inferred.update(new_inferences)
        
        # Add confidence adjustments for risk factors
        for entity_id in inferred:
            attrs = self.entity_attrs.get(entity_id, {})
            if 'risk_factor' in attrs and attrs['risk_factor'] > 0:
                for class_id, score in class_scores.items():
                    if attrs.get(f'increases_{class_id}', 0) > 0:
                        multiplier = 1 + (attrs['risk_factor'] * attrs[f'increases_{class_id}'])
                        class_scores[class_id] = min(0.99, score * multiplier)
                        reasoning_steps[f"risk_{entity_id}_{class_id}"] = (
                            f"Risk Factor: {self.entities[entity_id]} increases likelihood of "
                            f"{self.entities.get(class_id + self.symbol_offset, 'unknown')} by {multiplier:.1f}x"
                        )
        
        return inferred, reasoning_steps, confidences, dict(class_scores)

# ===========================
# 3. Advanced Neural-Symbolic Interface
# ===========================
class AdvancedNeuralSymbolicInterface(nn.Module):
    def __init__(self, hidden_dim, num_symbols, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_symbols = num_symbols
        self.num_classes = num_classes
        
        self.neural_to_symbol = nn.Linear(hidden_dim, num_symbols)
        self.symbol_to_class = nn.Parameter(torch.zeros(num_symbols, num_classes))
        self.threshold_base = nn.Parameter(torch.ones(1) * 0.5)
        self.threshold_scale = nn.Parameter(torch.ones(num_symbols) * 0.1)
        
    def forward(self, neural_repr):
        symbol_logits = self.neural_to_symbol(neural_repr)
        return symbol_logits
    
    def get_thresholds(self):
        return torch.clamp(self.threshold_base + self.threshold_scale, 0.1, 0.9)
    
    def neural_to_symbolic(self, neural_repr):
        symbol_logits = self.neural_to_symbol(neural_repr)
        symbol_probs = torch.sigmoid(symbol_logits)
        
        thresholds = self.get_thresholds()
        activations = (symbol_probs > thresholds).float()
        
        return activations, symbol_probs, symbol_logits
    
    def symbolic_to_neural_prediction(self, symbolic_activations, confidences=None):
        if confidences is None:
            class_scores = torch.matmul(symbolic_activations, self.symbol_to_class)
        else:
            conf_tensor = torch.zeros_like(symbolic_activations)
            for i, confs in enumerate(confidences):
                for symbol_id, conf in confs.items():
                    if isinstance(symbol_id, int) and symbol_id < conf_tensor.shape[1]:
                        conf_tensor[i, symbol_id] = conf
            weighted_activations = symbolic_activations * conf_tensor
            class_scores = torch.matmul(weighted_activations, self.symbol_to_class)
        
        return class_scores
    
    def set_symbol_to_class_mapping(self, symbol_to_class_dict):
        with torch.no_grad():
            for symbol_id, class_weights in symbol_to_class_dict.items():
                for class_id, weight in class_weights.items():
                    self.symbol_to_class[symbol_id, class_id] = weight

# ===========================
# 4. Advanced Metacognitive Control
# ===========================
class AdvancedMetacognitiveController:
    def __init__(self, neural_threshold=0.85, symbolic_threshold=0.75, learning_rate=0.01):
        self.neural_threshold = neural_threshold
        self.symbolic_threshold = symbolic_threshold
        self.learning_rate = learning_rate
        self.strategy_history = []
        self.correct_strategy_counts = {'neural': 0, 'symbolic': 0, 'hybrid': 0}
        
    def update_thresholds(self, neural_correct, symbolic_correct, strategy):
        if neural_correct != symbolic_correct:
            if neural_correct:
                self.neural_threshold = max(0.7, self.neural_threshold - self.learning_rate)
                self.symbolic_threshold = min(0.9, self.symbolic_threshold + self.learning_rate)
                self.correct_strategy_counts['neural'] += 1
            else:
                self.neural_threshold = min(0.9, self.neural_threshold + self.learning_rate)
                self.symbolic_threshold = max(0.7, self.symbolic_threshold - self.learning_rate)
                self.correct_strategy_counts['symbolic'] += 1
        elif neural_correct and symbolic_correct:
            if strategy == 'hybrid':
                self.correct_strategy_counts['hybrid'] += 1
        
    def decide_strategy(self, neural_conf, symbolic_conf, risk_level='medium'):
        neural_threshold = self.neural_threshold
        symbolic_threshold = self.symbolic_threshold
        
        if risk_level == 'high':
            neural_threshold += 0.1
            symbolic_threshold -= 0.1
        elif risk_level == 'low':
            neural_threshold -= 0.1
            symbolic_threshold += 0.1
            
        if neural_conf >= neural_threshold and symbolic_conf < symbolic_threshold:
            strategy = {
                'strategy': 'neural',
                'neural_weight': 1.0,
                'symbolic_weight': 0.0,
                'explanation': f'Using neural prediction (high confidence: {neural_conf:.2f})'
            }
        elif symbolic_conf >= symbolic_threshold and neural_conf < neural_threshold:
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
        
        self.strategy_history.append(strategy['strategy'])
        return strategy
    
    def get_strategy_stats(self):
        if not self.strategy_history:
            return {'neural': 0, 'symbolic': 0, 'hybrid': 0}
        return {
            'neural': self.strategy_history.count('neural') / len(self.strategy_history),
            'symbolic': self.strategy_history.count('symbolic') / len(self.strategy_history),
            'hybrid': self.strategy_history.count('hybrid') / len(self.strategy_history),
            'correct_neural': self.correct_strategy_counts['neural'],
            'correct_symbolic': self.correct_strategy_counts['symbolic'],
            'correct_hybrid': self.correct_strategy_counts['hybrid'],
        }

# ===========================
# 5. Enhanced NEXUS Integrated Model
# ===========================
class EnhancedNEXUSModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_symbols, symbol_names, class_names, 
                 embed_dim=128, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_symbols = num_symbols
        self.symbol_names = symbol_names
        self.class_names = class_names
        self.symbol_to_id = {name: i for i, name in enumerate(symbol_names)}
        self.device = device
        
        self = self.to(device)
        
        self.neural_model = AdvancedNeuralModel(
            input_dim=input_dim, 
            num_classes=num_classes,
            embed_dim=embed_dim
        ).to(device)
        
        self.knowledge_graph = EnhancedKnowledgeGraph()
        
        self.interface = AdvancedNeuralSymbolicInterface(
            hidden_dim=embed_dim,
            num_symbols=num_symbols,
            num_classes=num_classes
        ).to(device)
        
        self.metacognitive = AdvancedMetacognitiveController()
        
        self.eval_results = {
            'neural': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'symbolic': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'nexus': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []}
        }
        
        self.case_details = []
        
    def init_knowledge_graph(self):
        """Initialize the knowledge graph with entities and example relations"""
        kg = self.knowledge_graph
        
        # Add symptom entities (IDs: 0 ... num_symbols-1)
        for i, name in enumerate(self.symbol_names):
            kg.add_entity(i, name)
            
        # Add class entities with an offset (IDs: num_symbols ... num_symbols+num_classes-1)
        offset = self.num_symbols
        for i, name in enumerate(self.class_names):
            kg.add_entity(offset + i, name)
            
        # Add example relations (e.g., symptom_0 -> condition_0)
        if self.num_symbols > 0 and self.num_classes > 0:
            kg.add_relation(0, "indicates", offset + 0, weight=0.9)
            if self.num_symbols > 1:
                kg.add_rule([0, 1], offset + 1, confidence=0.85)
            
        return kg
    
    def forward(self, x):
        x = x.to(self.device)
        return self.neural_model(x)[0]
    
    def train_neural(self, dataloader, num_epochs=5, lr=0.001, scheduler=None, weight_decay=1e-5):
        self.neural_model.train()
        optimizer = torch.optim.AdamW(
            self.neural_model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        if scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs * len(dataloader)
            )
        elif scheduler == 'reduce':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2
            )
        
        criterion = nn.CrossEntropyLoss()
        
        epoch_stats = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs, _, _ = self.neural_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            epoch_stats.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
        return epoch_stats
    
    def diagnose(self, x, active_symptoms=None, risk_level='medium'):
        self.neural_model.eval()
        self.interface.eval()
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        with torch.no_grad():
            neural_logits, neural_repr, _ = self.neural_model(x)
            neural_probs = F.softmax(neural_logits, dim=1)
            neural_pred = torch.argmax(neural_probs, dim=1).item()
            neural_conf = neural_probs[0, neural_pred].item()
            
            symbolic_activations, similarities, _ = self.interface.neural_to_symbolic(neural_repr)
            
            if active_symptoms is not None:
                symptom_ids = [self.symbol_to_id[name] for name in active_symptoms if name in self.symbol_to_id]
            else:
                symptom_ids = torch.nonzero(symbolic_activations[0]).squeeze(-1).tolist()
                if not isinstance(symptom_ids, list):
                    symptom_ids = [symptom_ids]
            
            inferred_ids, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids)
            
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
            
            strategy = self.metacognitive.decide_strategy(neural_conf, symbolic_conf, risk_level)
            
            if strategy['strategy'] == 'neural':
                final_pred = neural_pred
                final_conf = neural_conf
            elif strategy['strategy'] == 'symbolic':
                final_pred = symbolic_pred
                final_conf = symbolic_conf
            else:
                combined_probs = (
                    strategy['neural_weight'] * neural_probs + 
                    strategy['symbolic_weight'] * symbolic_probs
                )
                final_pred = torch.argmax(combined_probs, dim=1).item()
                final_conf = combined_probs[0, final_pred].item()
        
        result = {
            'neural': {
                'prediction': neural_pred,
                'confidence': neural_conf,
                'class_name': self.class_names[neural_pred],
                'probabilities': neural_probs[0].cpu().numpy()
            },
            'symbolic': {
                'prediction': symbolic_pred,
                'confidence': symbolic_conf,
                'class_name': self.class_names[symbolic_pred],
                'reasoning_steps': reasoning_steps,
                'inferred_symbols': [self.symbol_names[i] for i in inferred_ids if i < len(self.symbol_names)],
                'active_symptoms': [self.symbol_names[i] for i in symptom_ids if i < len(self.symbol_names)],
                'class_scores': class_scores,
                'probabilities': symbolic_probs[0].cpu().numpy()
            },
            'nexus': {
                'prediction': final_pred,
                'confidence': final_conf,
                'class_name': self.class_names[final_pred],
                'strategy': strategy
            }
        }
        
        return result
    
    def evaluate(self, dataloader, symptom_dict=None, feedback=True):
        self.neural_model.eval()
        self.interface.eval()
        
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
        
        progress_bar = tqdm(dataloader, desc="Evaluating")
        self.case_details = []
        
        sample_index = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(progress_bar):
                batch_size = inputs.size(0)
                for j in range(batch_size):
                    active_symptoms = symptom_dict.get(sample_index, None) if symptom_dict else None
                    sample_input = inputs[j].unsqueeze(0)
                    true_label = labels[j].item()
                    
                    result = self.diagnose(sample_input, active_symptoms)
                    
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
                        'active_symptoms': active_symptoms
                    }
                    self.case_details.append(case_detail)
                    
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
                        
                    if feedback:
                        self.metacognitive.update_thresholds(
                            neural_correct, 
                            symbolic_correct,
                            result['nexus']['strategy']['strategy']
                        )
                    sample_index += 1
                    
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
        conf_str = f" (Confidence: {result['nexus']['confidence']:.2f})" if include_confidence else ""
        explanation = [f"Diagnosis: {result['nexus']['class_name']}{conf_str}"]
        explanation.append(f"Strategy: {result['nexus']['strategy']['strategy']}")
        explanation.append(f"Reason: {result['nexus']['strategy']['explanation']}")
        
        if detail_level == 'simple':
            return "\n".join(explanation)
        
        explanation.append("\nDetected Symptoms:")
        if 'active_symptoms' in result['symbolic'] and result['symbolic']['active_symptoms']:
            explanation.append(f"  {', '.join(result['symbolic']['active_symptoms'])}")
        else:
            explanation.append("  None detected")
        
        explanation.append("\nSymbolic Reasoning:")
        explanation.append(f"Identified concepts: {', '.join(result['symbolic']['inferred_symbols'])}")
        
        if detail_level == 'high' and result['symbolic']['reasoning_steps']:
            explanation.append("\nReasoning steps:")
            symptom_steps = []
            rule_steps = []
            other_steps = []
            
            for symbol_id, step in result['symbolic']['reasoning_steps'].items():
                if isinstance(symbol_id, (int, np.int64)) and symbol_id < len(self.symbol_names) + len(self.class_names):
                    if symbol_id < len(self.symbol_names):
                        symbol_name = self.symbol_names[symbol_id]
                    else:
                        symbol_name = self.class_names[symbol_id - len(self.symbol_names)]
                    formatted_step = f"- {symbol_name}: {step}"
                    if "Given" in step:
                        symptom_steps.append(formatted_step)
                    elif "Rule" in step:
                        rule_steps.append(formatted_step)
                    else:
                        other_steps.append(formatted_step)
                else:
                    other_steps.append(f"- {step}")
            
            if symptom_steps:
                explanation.append("Initial symptoms:")
                explanation.extend(symptom_steps)
            if rule_steps:
                explanation.append("\nApplied medical rules:")
                explanation.extend(rule_steps)
            if other_steps:
                explanation.append("\nOther reasoning:")
                explanation.extend(other_steps)
        
        neural_conf = f" (Confidence: {result['neural']['confidence']:.2f})" if include_confidence else ""
        symbolic_conf = f" (Confidence: {result['symbolic']['confidence']:.2f})" if include_confidence else ""
        
        explanation.append(f"\nNeural model prediction: {result['neural']['class_name']}{neural_conf}")
        explanation.append(f"Symbolic model prediction: {result['symbolic']['class_name']}{symbolic_conf}")
        
        if detail_level == 'high' and include_confidence:
            explanation.append("\nClass probabilities (Neural):")
            for i, prob in enumerate(result['neural']['probabilities']):
                explanation.append(f"  {self.class_names[i]}: {prob:.4f}")
            explanation.append("\nClass scores (Symbolic):")
            for i in range(len(self.class_names)):
                score = result['symbolic']['class_scores'].get(i, 0)
                explanation.append(f"  {self.class_names[i]}: {score:.4f}")
        
        return "\n".join(explanation)
    
    def visualize_results(self, output_prefix=None, save_figures=False, show_figures=True):
        if self.eval_results['neural']['confusion'] is None:
            print("No evaluation results to visualize. Run evaluate() first.")
            return
        
        models = ['neural', 'symbolic', 'nexus']
        titles = ['Neural Model', 'Symbolic Model', 'NEXUS Model']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # 1. Accuracy Comparison
        try:
            accuracies = [self.eval_results[model]['accuracy'] * 100 for model in models]
            plt.figure(figsize=(12, 6))
            bars = plt.bar(titles, accuracies, color=colors, alpha=0.8, width=0.6)
            plt.title('Accuracy Comparison', fontsize=16)
            plt.ylabel('Accuracy (%)', fontsize=14)
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1, 
                         f"{height:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.tight_layout()
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_accuracy.png", dpi=300, bbox_inches='tight')
            if show_figures:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Error in accuracy plot: {e}")
        
        # 2. Confusion Matrices
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Confusion Matrices (Normalized by Row)', fontsize=16, y=1.05)
            for i, (model, title, color) in enumerate(zip(models, titles, colors)):
                confusion = self.eval_results[model]['confusion']
                row_sums = confusion.sum(axis=1, keepdims=True)
                norm_confusion = np.where(row_sums == 0, 0, confusion / row_sums)
                cmap = "Blues" if i==0 else ("Greens" if i==1 else "Reds")
                sns.heatmap(norm_confusion, annot=True, fmt='.2f', cmap=cmap, 
                            xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[i])
                axes[i].set_title(title, fontsize=14)
                axes[i].set_ylabel('True Label' if i == 0 else '', fontsize=12)
                axes[i].set_xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_confusion.png", dpi=300, bbox_inches='tight')
            if show_figures:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Error in confusion matrix plot: {e}")
        
        # 3. Model Agreement Analysis
        try:
            agreement = self.eval_results['agreement_cases']
            labels = ['All Correct', 'Neural Only', 'Symbolic Only', 'NEXUS Better', 'All Wrong']
            values = [agreement['all_correct'], agreement['neural_only'], agreement['symbolic_only'], 
                      agreement['nexus_better'], agreement['all_wrong']]
            colors_agree = ['#27ae60', '#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
            total_cases = sum(values)
            percentages = [100 * v / total_cases for v in values] if total_cases > 0 else [0] * 5
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            bars = ax1.bar(labels, values, color=colors_agree, alpha=0.8)
            ax1.set_title('Model Agreement Analysis (Counts)', fontsize=16)
            ax1.set_ylabel('Number of Cases', fontsize=14)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                         f"{int(height)}", ha='center', va='bottom', fontsize=12)
            wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%', 
                                                colors=colors_agree, shadow=False, startangle=90)
            ax2.set_title('Model Agreement Analysis (Percentages)', fontsize=16)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
            plt.tight_layout()
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_agreement.png", dpi=300, bbox_inches='tight')
            if show_figures:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Error in agreement plot: {e}")
        
        # 4. Class-wise Performance
        try:
            f1_scores = np.zeros((self.num_classes, 3))
            for c in range(self.num_classes):
                for i, model in enumerate(models):
                    true_labels = np.array(self.eval_results[model]['true_labels'])
                    predictions = np.array(self.eval_results[model]['predictions'])
                    tp = np.sum((predictions == c) & (true_labels == c))
                    fp = np.sum((predictions == c) & (true_labels != c))
                    fn = np.sum((predictions != c) & (true_labels == c))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    f1_scores[c, i] = f1
            plt.figure(figsize=(14, 8))
            x = np.arange(self.num_classes)
            width = 0.25
            for i, (model, color) in enumerate(zip(models, titles, colors)):
                plt.bar(x + (i - 1) * width, f1_scores[:, i], width, color=color, label=titles[i], alpha=0.8)
            plt.xlabel('Class', fontsize=14)
            plt.ylabel('F1 Score', fontsize=14)
            plt.title('F1 Score by Class and Model', fontsize=16)
            plt.xticks(x, self.class_names, rotation=45, ha='right')
            plt.ylim(0, 1.0)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_f1_scores.png", dpi=300, bbox_inches='tight')
            if show_figures:
                plt.show()
            else:
                plt.close()
            class_results = []
            for c in range(self.num_classes):
                class_results.append([
                    self.class_names[c], 
                    f"{f1_scores[c, 0]:.3f}", 
                    f"{f1_scores[c, 1]:.3f}", 
                    f"{f1_scores[c, 2]:.3f}"
                ])
            print("\nClass-wise F1 Performance:")
            print(tabulate(class_results, headers=['Class', 'Neural F1', 'Symbolic F1', 'NEXUS F1'], tablefmt='grid'))
        except Exception as e:
            print(f"Error in class-wise performance visualization: {e}")
        
        # 5. Confidence Distribution
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Confidence Distribution by Model', fontsize=16, y=1.05)
            for i, (model, title, color) in enumerate(zip(models, titles, colors)):
                conf_values = self.eval_results[model]['confidence']
                correct = np.array(self.eval_results[model]['predictions']) == np.array(self.eval_results[model]['true_labels'])
                axes[i].hist([np.array(conf_values)[correct], np.array(conf_values)[~correct]], 
                             bins=20, stacked=True, color=[color, 'gray'], 
                             alpha=0.7, label=['Correct', 'Incorrect'])
                axes[i].set_title(title, fontsize=14)
                axes[i].set_xlabel('Confidence', fontsize=12)
                axes[i].set_ylabel('Count' if i == 0 else '', fontsize=12)
                axes[i].legend(fontsize=10)
                axes[i].grid(alpha=0.3)
            plt.tight_layout()
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_confidence.png", dpi=300, bbox_inches='tight')
            if show_figures:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Error in confidence distribution plot: {e}")
        
        # 6. Metacognitive Strategy Evolution
        try:
            strategy_stats = self.metacognitive.get_strategy_stats()
            strategy_counts = {
                'Neural': self.metacognitive.strategy_history.count('neural'),
                'Symbolic': self.metacognitive.strategy_history.count('symbolic'),
                'Hybrid': self.metacognitive.strategy_history.count('hybrid')
            }
            plt.figure(figsize=(12, 6))
            wedges, texts, autotexts = plt.pie(
                list(strategy_counts.values()), 
                labels=list(strategy_counts.keys()),
                autopct='%1.1f%%', 
                colors=['#3498db', '#2ecc71', '#9b59b6'],
                startangle=90
            )
            plt.title('Metacognitive Strategy Distribution', fontsize=16)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
            plt.tight_layout()
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_strategy_dist.png", dpi=300, bbox_inches='tight')
            if show_figures:
                plt.show()
            else:
                plt.close()
            print("\nMetacognitive Strategy Evolution:")
            print(f"Neural strategy used: {strategy_counts['Neural']} times ({strategy_counts['Neural']/sum(strategy_counts.values())*100:.1f}%)")
            print(f"Symbolic strategy used: {strategy_counts['Symbolic']} times ({strategy_counts['Symbolic']/sum(strategy_counts.values())*100:.1f}%)")
            print(f"Hybrid strategy used: {strategy_counts['Hybrid']} times ({strategy_counts['Hybrid']/sum(strategy_counts.values())*100:.1f}%)")
            if 'correct_neural' in strategy_stats:
                print(f"\nStrategy Effectiveness:")
                print(f"Correct with Neural: {strategy_stats['correct_neural']} cases")
                print(f"Correct with Symbolic: {strategy_stats['correct_symbolic']} cases")
                print(f"Correct with Hybrid: {strategy_stats['correct_hybrid']} cases")
        except Exception as e:
            print(f"Error in metacognitive strategy plot: {e}")
        
        summary = {
            'neural_accuracy': self.eval_results['neural']['accuracy'],
            'symbolic_accuracy': self.eval_results['symbolic']['accuracy'],
            'nexus_accuracy': self.eval_results['nexus']['accuracy'],
            'agreement_cases': self.eval_results['agreement_cases'],
            'metacognitive_stats': self.metacognitive.get_strategy_stats(),
            'class_f1_scores': {
                'neural': [f1_scores[c, 0] for c in range(self.num_classes)],
                'symbolic': [f1_scores[c, 1] for c in range(self.num_classes)],
                'nexus': [f1_scores[c, 2] for c in range(self.num_classes)]
            }
        }
        
        return summary

    def export_results(self, filename):
        try:
            if not self.case_details:
                print("No case details available. Run evaluate() first.")
                return False
                
            df = pd.DataFrame(self.case_details)
            df['neural_correct'] = df['neural_pred'] == df['true_label']
            df['symbolic_correct'] = df['symbolic_pred'] == df['true_label']
            df['nexus_correct'] = df['nexus_pred'] == df['true_label']
            df['nexus_improved'] = ((~df['neural_correct'] | ~df['symbolic_correct']) & df['nexus_correct'])
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

# ===========================
# 6. Experiment Runner Function
# ===========================
def run_nexus_experiment_real_data(dataset_name, max_samples, num_epochs, batch_size, learning_rate, output_dir, device, random_state):
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split='train')
    if max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    # Dynamically detect feature and label columns
    data_dict = dataset[0]
    feature_key = next((k for k in data_dict if k in ['features', 'feature', 'text', 'data']), None)
    label_key = next((k for k in data_dict if k in ['label', 'labels', 'target']), None)
    if not feature_key or not label_key:
        raise ValueError(f"Dataset must contain feature and label columns. Found: {list(data_dict.keys())}")
    
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
    num_symbols = 10
    symbol_names = [f"symptom_{i}" for i in range(num_symbols)]
    class_names = [f"condition_{i}" for i in range(num_classes)]
    
    model = EnhancedNEXUSModel(input_dim=input_dim, num_classes=num_classes, num_symbols=num_symbols,
                               symbol_names=symbol_names, class_names=class_names, embed_dim=128, device=device)
    model.knowledge_graph.symbol_offset = num_symbols
    model.knowledge_graph.num_classes = num_classes
    model.init_knowledge_graph()
    
    print("Training neural component...")
    model.train_neural(train_loader, num_epochs=num_epochs, lr=learning_rate)
    
    print("Evaluating model on test set...")
    test_results = model.evaluate(test_loader, symptom_dict=None, feedback=False)
    
    os.makedirs(output_dir, exist_ok=True)
    model.export_results(os.path.join(output_dir, "evaluation_results.csv"))
    
    return {'model': model, 'test_results': test_results}

# ===========================
# 7. Main Execution Block
# ===========================
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Run NEXUS experiment with real medical data")
    parser.add_argument("--dataset", type=str, default="medical-dataset/processed-dataset", 
                        help="Hugging Face dataset name")
    parser.add_argument("--samples", type=int, default=10000, help="Maximum number of patient cases")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NEXUS Transformer for Medical Diagnosis with Real Data")
    print(f"Analyzing up to {args.samples} real patient cases from {args.dataset}")
    print("=" * 80)
    
    start_time = time.time()
    
    experiment_results = run_nexus_experiment_real_data(
        dataset_name=args.dataset,
        max_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
        device=args.device,
        random_state=args.seed
    )
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"Experiment completed in {(end_time - start_time) / 60:.2f} minutes")
    print("=" * 80)
    
    model = experiment_results['model']
    test_results = experiment_results['test_results']
    
    print("\nFinal Comparative Summary:")
    print("-" * 40)
    print(f"Neural Model Accuracy: {test_results['neural']['accuracy']*100:.2f}%")
    print(f"Symbolic Model Accuracy: {test_results['symbolic']['accuracy']*100:.2f}%")
    print(f"NEXUS Model Accuracy: {test_results['nexus']['accuracy']*100:.2f}%")
    
    neural_acc = test_results['neural']['accuracy']
    symbolic_acc = test_results['symbolic']['accuracy']
    nexus_acc = test_results['nexus']['accuracy']
    
    best_component = max(neural_acc, symbolic_acc)
    improvement = (nexus_acc - best_component) * 100
    
    print(f"\nNEXUS improvement over best component: {improvement:.2f}%")
    
    agreement = test_results['agreement_cases']
    total = sum(agreement.values())
    
    print("\nAgreement Analysis:")
    print(f"All models correct: {agreement['all_correct']} cases ({100*agreement['all_correct']/total:.1f}%)")
    print(f"Neural only correct: {agreement['neural_only']} cases ({100*agreement['neural_only']/total:.1f}%)")
    print(f"Symbolic only correct: {agreement['symbolic_only']} cases ({100*agreement['symbolic_only']/total:.1f}%)")
    print(f"NEXUS better than components: {agreement['nexus_better']} cases ({100*agreement['nexus_better']/total:.1f}%)")
    print(f"All models wrong: {agreement['all_wrong']} cases ({100*agreement['all_wrong']/total:.1f}%)")
    
    strategy_stats = model.metacognitive.get_strategy_stats()
    
    print("\nMetacognitive Strategy Usage:")
    if 'neural' in strategy_stats:
        print(f"Neural strategy: {strategy_stats['neural']*100:.1f}%")
        print(f"Symbolic strategy: {strategy_stats['symbolic']*100:.1f}%")
        print(f"Hybrid strategy: {strategy_stats['hybrid']*100:.1f}%")
    
    print("\nConclusion:")
    if nexus_acc > best_component:
        print("NEXUS successfully improved over both neural and symbolic components!")
    elif nexus_acc == best_component:
        print("NEXUS performed equally to the best component.")
    else:
        print("NEXUS did not improve over the best component in this experiment.")
        
    print("\nComplete results saved to:", args.output)
    print("=" * 80)