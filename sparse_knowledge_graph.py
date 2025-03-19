import numpy as np
import torch
import os
import json
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from collections import defaultdict
import scipy.sparse as sp
import networkx as nx
import pickle
from tqdm import tqdm

class SparseKnowledgeGraph:
    """
    Scalable, memory-efficient implementation of a knowledge graph using sparse representations
    that can handle millions of entities and relationships.
    """
    def __init__(self, name: str = "graph", storage_dir: Optional[str] = None):
        # Graph metadata
        self.name = name
        self.storage_dir = storage_dir
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        # Entity storage
        self.entities = {}                 # entity_id -> name
        self.entity_attrs = {}             # entity_id -> {attribute: value}
        self.entity_to_id = {}             # name -> entity_id
        self.id_counter = 0                # For generating unique IDs
        
        # Hierarchical indexing
        self.hierarchy = defaultdict(set)  # entity_id -> set of parent entity_ids
        self.descendants = defaultdict(set)  # entity_id -> set of child entity_ids
        self.type_index = defaultdict(set)  # type -> set of entity_ids of that type
        
        # Sparse relation representation
        self.relation_types = set()        # Set of all relation types
        self.relation_type_to_id = {}      # relation_type -> relation_id
        self.id_to_relation_type = {}      # relation_id -> relation_type
        self.relation_id_counter = 0       # For generating unique relation IDs
        
        # Adjacency matrices for relationships (sparse)
        self.relation_matrices = {}        # relation_type -> scipy.sparse.csr_matrix
        self.relation_weights = {}         # (source_id, relation_type, target_id) -> weight
        
        # Rule storage
        self.rules = []                    # (premise_ids, conclusion_id, confidence)
        self.rule_index = defaultdict(set) # conclusion_id -> set of rule indices concluding this entity
        
        # Configuration
        self.symbol_offset = 0             # Should be set externally
        self.num_classes = 0               # Should be set externally
        
        # Memory optimization flags
        self.use_sparse_matrices = True    # Whether to use sparse matrices for relations
        self.use_disk_offloading = False   # Whether to offload large matrices to disk
        self.chunk_size = 10000            # Size of chunks for distributed processing
        
    def add_entity(self, entity_id: int, name: str, attributes: Optional[Dict] = None) -> 'SparseKnowledgeGraph':
        """Add an entity to the knowledge graph with optional attributes"""
        if entity_id in self.entities:
            # Update existing entity
            self.entities[entity_id] = name
            if attributes:
                if entity_id not in self.entity_attrs:
                    self.entity_attrs[entity_id] = {}
                self.entity_attrs[entity_id].update(attributes)
        else:
            # Add new entity
            self.entities[entity_id] = name
            if attributes:
                self.entity_attrs[entity_id] = attributes
                # Index by type if available
                if 'type' in attributes:
                    self.type_index[attributes['type']].add(entity_id)
                    
            # Update inverse mapping
            self.entity_to_id[name] = entity_id
            
            # Update ID counter if needed
            if entity_id >= self.id_counter:
                self.id_counter = entity_id + 1
                
        return self
    
    def add_entity_by_name(self, name: str, attributes: Optional[Dict] = None) -> int:
        """Add entity by name and return its ID"""
        if name in self.entity_to_id:
            entity_id = self.entity_to_id[name]
            # Update attributes if provided
            if attributes and entity_id in self.entity_attrs:
                self.entity_attrs[entity_id].update(attributes)
            elif attributes:
                self.entity_attrs[entity_id] = attributes
            return entity_id
        else:
            # Create new entity
            entity_id = self.id_counter
            self.id_counter += 1
            self.add_entity(entity_id, name, attributes)
            return entity_id
    
    def _ensure_relation_matrix(self, relation_type: str) -> None:
        """Ensure the relation matrix exists for the given relation type"""
        if relation_type not in self.relation_matrices:
            # Add to relation type registry
            if relation_type not in self.relation_type_to_id:
                relation_id = self.relation_id_counter
                self.relation_id_counter += 1
                self.relation_type_to_id[relation_type] = relation_id
                self.id_to_relation_type[relation_id] = relation_type
                self.relation_types.add(relation_type)
                
            # Create sparse matrix for this relation
            max_id = max(self.entities.keys()) if self.entities else 0
            size = max(100, max_id + 1)  # Minimum size of 100 for small graphs
            
            if self.use_sparse_matrices:
                # Create empty sparse matrix
                self.relation_matrices[relation_type] = sp.lil_matrix((size, size))
            else:
                # Use dictionary representation for very sparse relationships
                self.relation_matrices[relation_type] = {}
    
    def _resize_relation_matrix(self, relation_type: str, min_size: int) -> None:
        """Resize the relation matrix to accommodate larger entity IDs"""
        if relation_type not in self.relation_matrices:
            self._ensure_relation_matrix(relation_type)
            return
        
        current_matrix = self.relation_matrices[relation_type]
        
        if self.use_sparse_matrices:
            current_size = current_matrix.shape[0]
            if current_size <= min_size:
                # Calculate new size with padding for future growth
                new_size = max(current_size * 2, min_size + 100)
                
                # Create larger matrix
                new_matrix = sp.lil_matrix((new_size, new_size))
                
                # Copy data from old matrix
                new_matrix[:current_size, :current_size] = current_matrix
                
                # Replace with resized matrix
                self.relation_matrices[relation_type] = new_matrix
        
    def add_relation(self, source_id: int, relation_type: str, target_id: int, weight: float = 1.0) -> 'SparseKnowledgeGraph':
        """Add a relation between two entities with a weight"""
        # Ensure relation matrix exists
        self._ensure_relation_matrix(relation_type)
        
        # Make sure matrix is large enough
        max_id = max(source_id, target_id)
        if self.use_sparse_matrices:
            current_size = self.relation_matrices[relation_type].shape[0]
            if current_size <= max_id:
                self._resize_relation_matrix(relation_type, max_id)
        
        # Add the relation
        if self.use_sparse_matrices:
            self.relation_matrices[relation_type][source_id, target_id] = weight
        else:
            if relation_type not in self.relation_matrices:
                self.relation_matrices[relation_type] = {}
            self.relation_matrices[relation_type][(source_id, target_id)] = weight
            
        # Store the weight
        self.relation_weights[(source_id, relation_type, target_id)] = weight
        
        return self
    
    def add_rule(self, premise_ids: List[int], conclusion_id: int, confidence: float = 1.0) -> 'SparseKnowledgeGraph':
        """Add a logical rule with a confidence score"""
        # Add the rule
        rule_index = len(self.rules)
        self.rules.append((premise_ids, conclusion_id, confidence))
        
        # Update rule index for faster lookup
        self.rule_index[conclusion_id].add(rule_index)
        
        return self
    
    def add_hierarchy(self, child_id: int, parent_id: int) -> 'SparseKnowledgeGraph':
        """Add hierarchical relationship (child is a type of parent)"""
        self.hierarchy[child_id].add(parent_id)
        self.descendants[parent_id].add(child_id)
        
        # Propagate to all ancestors
        ancestors_to_process = list(self.hierarchy[parent_id])
        for ancestor in ancestors_to_process:
            self.descendants[ancestor].add(child_id)
            
        return self
    
    def get_ancestors(self, entity_id: int) -> Set[int]:
        """Get all ancestors of an entity in the hierarchy"""
        ancestors = set()
        to_process = list(self.hierarchy[entity_id])
        
        while to_process:
            parent = to_process.pop()
            ancestors.add(parent)
            # Add this parent's ancestors to processing queue
            for grandparent in self.hierarchy[parent]:
                if grandparent not in ancestors:
                    to_process.append(grandparent)
            
        return ancestors
    
    def get_descendants(self, entity_id: int) -> Set[int]:
        """Get all descendants of an entity in the hierarchy (pre-computed)"""
        return self.descendants[entity_id]
    
    def get_entities_by_type(self, entity_type: str) -> Set[int]:
        """Get all entities of a specific type"""
        return self.type_index[entity_type]
    
    def get_connected_entities(self, entity_id: int, relation_type: str) -> Dict[int, float]:
        """
        Get all entities connected to the given entity by the specified relation type
        
        Returns a dictionary mapping target_id -> relationship weight
        """
        if relation_type not in self.relation_matrices:
            return {}
            
        if self.use_sparse_matrices:
            matrix = self.relation_matrices[relation_type]
            # Check if entity_id is within matrix bounds
            if entity_id >= matrix.shape[0]:
                return {}
                
            # Get row for source entity
            row = matrix[entity_id].tocsr()
            
            # Extract non-zero entries
            indices = row.indices
            data = row.data
            
            # Create dictionary of target_id -> weight
            connections = {int(idx): float(weight) for idx, weight in zip(indices, data)}
            return connections
        else:
            # For dictionary representation
            connections = {}
            for (src, tgt), weight in self.relation_matrices[relation_type].items():
                if src == entity_id:
                    connections[tgt] = weight
            return connections
    
    def get_entities_with_relation_to(self, entity_id: int, relation_type: str) -> Dict[int, float]:
        """
        Get all entities that have the specified relation to the given entity
        
        Returns a dictionary mapping source_id -> relationship weight
        """
        if relation_type not in self.relation_matrices:
            return {}
            
        if self.use_sparse_matrices:
            matrix = self.relation_matrices[relation_type]
            # Check if entity_id is within matrix bounds
            if entity_id >= matrix.shape[0]:
                return {}
                
            # Get column for target entity
            col = matrix[:,entity_id].tocsc()
            
            # Extract non-zero entries
            indices = col.indices
            data = col.data
            
            # Create dictionary of source_id -> weight
            connections = {int(idx): float(weight) for idx, weight in zip(indices, data)}
            return connections
        else:
            # For dictionary representation
            connections = {}
            for (src, tgt), weight in self.relation_matrices[relation_type].items():
                if tgt == entity_id:
                    connections[src] = weight
            return connections
    
    def reason(self, active_entities: List[int], max_hops: int = 3) -> Tuple[Set[int], Dict, Dict, Dict]:
        """
        Apply enhanced reasoning to derive new knowledge with support for large graphs
        
        Args:
            active_entities: List of active entity IDs
            max_hops: Maximum number of reasoning hops
            
        Returns:
            inferred: Set of inferred entity IDs
            reasoning_steps: Dictionary of reasoning steps for each entity
            confidences: Dictionary of confidence values for each entity
            class_scores: Dictionary of confidence scores for each class
        """
        # Initialize with active entities
        inferred = set(active_entities)
        
        # Add hierarchical parents (ancestors)
        for entity in list(active_entities):
            inferred.update(self.get_ancestors(entity))
            
        reasoning_steps = {}
        confidences = {}
        
        # Default class scores
        class_scores = defaultdict(float)
        
        # Process active entities in smaller batches if there are many
        for i in range(0, len(active_entities), self.chunk_size):
            batch = active_entities[i:i+self.chunk_size]
            
            # Initialize reasoning steps and confidences for active entities
            for entity_id in batch:
                if entity_id in self.entities:
                    reasoning_steps[entity_id] = f"Given: {self.entities[entity_id]}"
                    confidences[entity_id] = 1.0
        
        # Add reasoning steps for ancestor entities
        ancestor_entities = inferred - set(active_entities)
        for i in range(0, len(list(ancestor_entities)), self.chunk_size):
            batch = list(ancestor_entities)[i:i+self.chunk_size]
            
            for entity_id in batch:
                if entity_id in self.entities:
                    for child in active_entities:
                        if entity_id in self.get_ancestors(child):
                            reasoning_steps[entity_id] = f"Hierarchical: {self.entities[child]} is a type of {self.entities[entity_id]}"
                            confidences[entity_id] = 0.95
                            break
        
        # Multi-hop reasoning with batch processing
        for _ in range(max_hops):
            new_inferences = set()
            
            # Process existing inferred entities in batches
            for i in range(0, len(list(inferred)), self.chunk_size):
                inferred_batch = list(inferred)[i:i+self.chunk_size]
                
                # Apply relations for this batch
                for relation_type in self.relation_types:
                    for source_id in inferred_batch:
                        # Get all entities connected to this source
                        connected = self.get_connected_entities(source_id, relation_type)
                        
                        for target_id, weight in connected.items():
                            if target_id not in inferred:
                                new_inferences.add(target_id)
                                if source_id in self.entities and target_id in self.entities:
                                    step = f"{self.entities[source_id]} --{relation_type}--> {self.entities[target_id]}"
                                    reasoning_steps[target_id] = step
                                    confidences[target_id] = weight * confidences.get(source_id, 1.0)
                                    
                                    # Update class scores if this is a class entity
                                    if target_id >= self.symbol_offset and target_id < self.symbol_offset + self.num_classes:
                                        key = target_id - self.symbol_offset
                                        class_scores[key] = max(class_scores[key], confidences[target_id])
            
            # Apply rules in batches
            # First, find applicable rules
            applicable_rules = []
            for rule_idx, (premise_ids, conclusion_id, confidence) in enumerate(self.rules):
                if all(p_id in inferred for p_id in premise_ids) and conclusion_id not in inferred:
                    applicable_rules.append((rule_idx, premise_ids, conclusion_id, confidence))
            
            # Process applicable rules in batches
            for i in range(0, len(applicable_rules), self.chunk_size):
                rule_batch = applicable_rules[i:i+self.chunk_size]
                
                for _, premise_ids, conclusion_id, confidence in rule_batch:
                    new_inferences.add(conclusion_id)
                    
                    # Generate reasoning step
                    if all(p_id in self.entities for p_id in premise_ids) and conclusion_id in self.entities:
                        premises = [self.entities[p_id] for p_id in premise_ids]
                        step = f"Rule: IF {' AND '.join(premises)} THEN {self.entities[conclusion_id]}"
                        reasoning_steps[conclusion_id] = step
                        
                        # Calculate confidence
                        premise_conf = min([confidences.get(p_id, 1.0) for p_id in premise_ids])
                        rule_conf = confidence * premise_conf
                        confidences[conclusion_id] = rule_conf
                        
                        # Update class scores if this is a class entity
                        if conclusion_id >= self.symbol_offset and conclusion_id < self.symbol_offset + self.num_classes:
                            key = conclusion_id - self.symbol_offset
                            class_scores[key] = max(class_scores[key], rule_conf)
            
            # If no new inferences were made, stop
            if not new_inferences:
                break
                
            # Add new inferences
            inferred.update(new_inferences)
        
        # Handle confidence adjustments for risk factors in batches
        all_inferred = list(inferred)
        for i in range(0, len(all_inferred), self.chunk_size):
            entity_batch = all_inferred[i:i+self.chunk_size]
            
            for entity_id in entity_batch:
                attrs = self.entity_attrs.get(entity_id, {})
                if 'risk_factor' in attrs and attrs['risk_factor'] > 0:
                    for class_id, score in class_scores.items():
                        if attrs.get(f'increases_{class_id}', 0) > 0:
                            multiplier = 1 + (attrs['risk_factor'] * attrs[f'increases_{class_id}'])
                            class_scores[class_id] = min(0.99, score * multiplier)
                            
                            # Create reasoning step for this risk factor
                            if entity_id in self.entities and class_id + self.symbol_offset in self.entities:
                                step_id = f"risk_{entity_id}_{class_id}"
                                reasoning_steps[step_id] = (
                                    f"Risk Factor: {self.entities[entity_id]} increases likelihood of "
                                    f"{self.entities[class_id + self.symbol_offset]} by {multiplier:.1f}x"
                                )
        
        return inferred, reasoning_steps, confidences, dict(class_scores)
    
    def save_to_disk(self, directory: Optional[str] = None) -> None:
        """Save the knowledge graph to disk"""
        # Use provided directory or default
        save_dir = directory or self.storage_dir
        if not save_dir:
            raise ValueError("No storage directory specified")
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save metadata
        metadata = {
            'name': self.name,
            'id_counter': self.id_counter,
            'relation_id_counter': self.relation_id_counter,
            'symbol_offset': self.symbol_offset,
            'num_classes': self.num_classes,
            'use_sparse_matrices': self.use_sparse_matrices,
            'use_disk_offloading': self.use_disk_offloading,
            'chunk_size': self.chunk_size
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
        # Save entities and attributes
        entity_data = {
            'entities': self.entities,
            'entity_attrs': self.entity_attrs,
            'entity_to_id': self.entity_to_id
        }
        
        with open(os.path.join(save_dir, 'entities.pkl'), 'wb') as f:
            pickle.dump(entity_data, f)
            
        # Save hierarchy
        hierarchy_data = {
            'hierarchy': dict(self.hierarchy),
            'descendants': dict(self.descendants),
            'type_index': dict(self.type_index)
        }
        
        with open(os.path.join(save_dir, 'hierarchy.pkl'), 'wb') as f:
            pickle.dump(hierarchy_data, f)
            
        # Save relation types
        relation_type_data = {
            'relation_types': list(self.relation_types),
            'relation_type_to_id': self.relation_type_to_id,
            'id_to_relation_type': self.id_to_relation_type
        }
        
        with open(os.path.join(save_dir, 'relation_types.pkl'), 'wb') as f:
            pickle.dump(relation_type_data, f)
            
        # Save relation weights
        with open(os.path.join(save_dir, 'relation_weights.pkl'), 'wb') as f:
            pickle.dump(self.relation_weights, f)
            
        # Save relation matrices (potentially large)
        relation_dir = os.path.join(save_dir, 'relations')
        if not os.path.exists(relation_dir):
            os.makedirs(relation_dir)
            
        for relation_type, matrix in self.relation_matrices.items():
            # For sparse matrices, use specialized formats
            if self.use_sparse_matrices:
                # Convert to CSR for efficient storage
                if not isinstance(matrix, sp.csr_matrix):
                    matrix = matrix.tocsr()
                    
                sp.save_npz(os.path.join(relation_dir, f"{relation_type}.npz"), matrix)
            else:
                # For dictionary representation
                with open(os.path.join(relation_dir, f"{relation_type}.pkl"), 'wb') as f:
                    pickle.dump(matrix, f)
        
        # Save rules
        rules_data = {
            'rules': self.rules,
            'rule_index': dict(self.rule_index)
        }
        
        with open(os.path.join(save_dir, 'rules.pkl'), 'wb') as f:
            pickle.dump(rules_data, f)
    
    @classmethod
    def load_from_disk(cls, directory: str) -> 'SparseKnowledgeGraph':
        """Load knowledge graph from disk"""
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
            
        # Load metadata
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        # Create new instance
        graph = cls(name=metadata['name'], storage_dir=directory)
        graph.id_counter = metadata['id_counter']
        graph.relation_id_counter = metadata['relation_id_counter']
        graph.symbol_offset = metadata['symbol_offset']
        graph.num_classes = metadata['num_classes']
        graph.use_sparse_matrices = metadata['use_sparse_matrices']
        graph.use_disk_offloading = metadata['use_disk_offloading']
        graph.chunk_size = metadata['chunk_size']
        
        # Load entities and attributes
        with open(os.path.join(directory, 'entities.pkl'), 'rb') as f:
            entity_data = pickle.load(f)
            graph.entities = entity_data['entities']
            graph.entity_attrs = entity_data['entity_attrs']
            graph.entity_to_id = entity_data['entity_to_id']
            
        # Load hierarchy
        with open(os.path.join(directory, 'hierarchy.pkl'), 'rb') as f:
            hierarchy_data = pickle.load(f)
            graph.hierarchy = defaultdict(set, hierarchy_data['hierarchy'])
            graph.descendants = defaultdict(set, hierarchy_data['descendants'])
            graph.type_index = defaultdict(set, hierarchy_data['type_index'])
            
        # Load relation types
        with open(os.path.join(directory, 'relation_types.pkl'), 'rb') as f:
            relation_type_data = pickle.load(f)
            graph.relation_types = set(relation_type_data['relation_types'])
            graph.relation_type_to_id = relation_type_data['relation_type_to_id']
            graph.id_to_relation_type = relation_type_data['id_to_relation_type']
            
        # Load relation weights
        with open(os.path.join(directory, 'relation_weights.pkl'), 'rb') as f:
            graph.relation_weights = pickle.load(f)
            
        # Load relation matrices
        relation_dir = os.path.join(directory, 'relations')
        if os.path.exists(relation_dir):
            for relation_type in graph.relation_types:
                # For sparse matrices
                if graph.use_sparse_matrices:
                    matrix_path = os.path.join(relation_dir, f"{relation_type}.npz")
                    if os.path.exists(matrix_path):
                        graph.relation_matrices[relation_type] = sp.load_npz(matrix_path)
                else:
                    # For dictionary representation
                    dict_path = os.path.join(relation_dir, f"{relation_type}.pkl")
                    if os.path.exists(dict_path):
                        with open(dict_path, 'rb') as f:
                            graph.relation_matrices[relation_type] = pickle.load(f)
                            
        # Load rules
        with open(os.path.join(directory, 'rules.pkl'), 'rb') as f:
            rules_data = pickle.load(f)
            graph.rules = rules_data['rules']
            graph.rule_index = defaultdict(set, rules_data['rule_index'])
            
        return graph
    
    def convert_to_networkx(self) -> nx.MultiDiGraph:
        """Convert to NetworkX graph for visualization and analysis"""
        G = nx.MultiDiGraph()
        
        # Add entities as nodes
        for entity_id, name in self.entities.items():
            attrs = self.entity_attrs.get(entity_id, {})
            G.add_node(entity_id, name=name, **attrs)
            
        # Add relations as edges
        for (source_id, relation_type, target_id), weight in self.relation_weights.items():
            G.add_edge(source_id, target_id, key=relation_type, weight=weight, type=relation_type)
            
        return G
    
    def create_subgraph(self, entity_ids: List[int]) -> 'SparseKnowledgeGraph':
        """Create a subgraph containing only the specified entities and their relations"""
        subgraph = SparseKnowledgeGraph(name=f"{self.name}_sub")
        subgraph.use_sparse_matrices = self.use_sparse_matrices
        subgraph.symbol_offset = self.symbol_offset
        subgraph.num_classes = self.num_classes
        
        # Add entities
        for entity_id in entity_ids:
            if entity_id in self.entities:
                attrs = self.entity_attrs.get(entity_id, {})
                subgraph.add_entity(entity_id, self.entities[entity_id], attrs)
                
        # Add relations between these entities
        for (source_id, relation_type, target_id), weight in self.relation_weights.items():
            if source_id in entity_ids and target_id in entity_ids:
                subgraph.add_relation(source_id, relation_type, target_id, weight)
                
        # Add hierarchy relationships
        for child_id in entity_ids:
            for parent_id in self.hierarchy[child_id]:
                if parent_id in entity_ids:
                    subgraph.add_hierarchy(child_id, parent_id)
                    
        # Add rules that only involve these entities
        for premise_ids, conclusion_id, confidence in self.rules:
            if all(p_id in entity_ids for p_id in premise_ids) and conclusion_id in entity_ids:
                subgraph.add_rule(premise_ids, conclusion_id, confidence)
                
        return subgraph
    
    def merge(self, other: 'SparseKnowledgeGraph', prefix1: str = '', prefix2: str = '') -> 'SparseKnowledgeGraph':
        """
        Merge two knowledge graphs, handling entity ID conflicts
        
        Args:
            other: Another knowledge graph to merge with this one
            prefix1: Prefix to add to entity names from this graph
            prefix2: Prefix to add to entity names from the other graph
            
        Returns:
            A new merged knowledge graph
        """
        merged = SparseKnowledgeGraph(name=f"{self.name}_{other.name}_merged")
        merged.use_sparse_matrices = self.use_sparse_matrices
        
        # Keep track of remapped entity IDs
        id_mapping1 = {}  # original_id -> new_id
        id_mapping2 = {}  # original_id -> new_id
        
        # Add entities from first graph with prefix
        for entity_id, name in self.entities.items():
            new_name = f"{prefix1}{name}" if prefix1 else name
            attrs = self.entity_attrs.get(entity_id, {})
            
            new_id = merged.add_entity_by_name(new_name, attrs)
            id_mapping1[entity_id] = new_id
            
        # Add entities from second graph with prefix
        for entity_id, name in other.entities.items():
            new_name = f"{prefix2}{name}" if prefix2 else name
            attrs = other.entity_attrs.get(entity_id, {})
            
            new_id = merged.add_entity_by_name(new_name, attrs)
            id_mapping2[entity_id] = new_id
            
        # Add relations from first graph
        for (source_id, relation_type, target_id), weight in self.relation_weights.items():
            if source_id in id_mapping1 and target_id in id_mapping1:
                merged.add_relation(
                    id_mapping1[source_id], 
                    relation_type, 
                    id_mapping1[target_id], 
                    weight
                )
                
        # Add relations from second graph
        for (source_id, relation_type, target_id), weight in other.relation_weights.items():
            if source_id in id_mapping2 and target_id in id_mapping2:
                merged.add_relation(
                    id_mapping2[source_id], 
                    relation_type, 
                    id_mapping2[target_id], 
                    weight
                )
                
        # Add hierarchy from first graph
        for child_id, parents in self.hierarchy.items():
            if child_id in id_mapping1:
                for parent_id in parents:
                    if parent_id in id_mapping1:
                        merged.add_hierarchy(id_mapping1[child_id], id_mapping1[parent_id])
                        
        # Add hierarchy from second graph
        for child_id, parents in other.hierarchy.items():
            if child_id in id_mapping2:
                for parent_id in parents:
                    if parent_id in id_mapping2:
                        merged.add_hierarchy(id_mapping2[child_id], id_mapping2[parent_id])
                        
        # Add rules from first graph
        for premise_ids, conclusion_id, confidence in self.rules:
            mapped_premises = [id_mapping1[p_id] for p_id in premise_ids if p_id in id_mapping1]
            if len(mapped_premises) == len(premise_ids) and conclusion_id in id_mapping1:
                merged.add_rule(mapped_premises, id_mapping1[conclusion_id], confidence)
                
        # Add rules from second graph
        for premise_ids, conclusion_id, confidence in other.rules:
            mapped_premises = [id_mapping2[p_id] for p_id in premise_ids if p_id in id_mapping2]
            if len(mapped_premises) == len(premise_ids) and conclusion_id in id_mapping2:
                merged.add_rule(mapped_premises, id_mapping2[conclusion_id], confidence)
                
        return merged

class DistributedKnowledgeGraph:
    """
    Distributed knowledge graph for extremely large scale processing
    that partitions the graph across multiple storage nodes
    """
    def __init__(
        self, 
        name: str = "distributed_graph", 
        storage_dir: Optional[str] = None,
        num_partitions: int = 4
    ):
        self.name = name
        self.storage_dir = storage_dir
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        self.num_partitions = num_partitions
        self.partition_graphs = []
        
        # Create partition subdirectories
        if storage_dir:
            for i in range(num_partitions):
                partition_dir = os.path.join(storage_dir, f"partition_{i}")
                if not os.path.exists(partition_dir):
                    os.makedirs(partition_dir)
                    
                # Create a sparse graph for each partition
                self.partition_graphs.append(
                    SparseKnowledgeGraph(
                        name=f"{name}_partition_{i}",
                        storage_dir=partition_dir
                    )
                )
        else:
            # In-memory partitions
            for i in range(num_partitions):
                self.partition_graphs.append(
                    SparseKnowledgeGraph(name=f"{name}_partition_{i}")
                )
                
        # Entity partitioning
        self.entity_partition_map = {}  # entity_id -> partition_id
        
        # Shared configuration
        self.symbol_offset = 0
        self.num_classes = 0
        
    def _get_partition_id(self, entity_id: int) -> int:
        """Determine which partition an entity belongs to"""
        if entity_id in self.entity_partition_map:
            return self.entity_partition_map[entity_id]
        else:
            # Assign to a partition based on hash
            partition_id = hash(entity_id) % self.num_partitions
            self.entity_partition_map[entity_id] = partition_id
            return partition_id
    
    def add_entity(self, entity_id: int, name: str, attributes: Optional[Dict] = None) -> 'DistributedKnowledgeGraph':
        """Add an entity to the appropriate partition"""
        partition_id = self._get_partition_id(entity_id)
        self.partition_graphs[partition_id].add_entity(entity_id, name, attributes)
        return self
    
    def add_relation(self, source_id: int, relation_type: str, target_id: int, weight: float = 1.0) -> 'DistributedKnowledgeGraph':
        """Add a relation to the appropriate partitions"""
        source_partition = self._get_partition_id(source_id)
        target_partition = self._get_partition_id(target_id)
        
        # Add to source partition
        self.partition_graphs[source_partition].add_relation(source_id, relation_type, target_id, weight)
        
        # If target is in a different partition, add there too for bidirectional access
        if source_partition != target_partition:
            self.partition_graphs[target_partition].add_relation(source_id, relation_type, target_id, weight)
            
        return self
    
    def add_rule(self, premise_ids: List[int], conclusion_id: int, confidence: float = 1.0) -> 'DistributedKnowledgeGraph':
        """Add a rule to the appropriate partitions"""
        # Add to conclusion's partition
        conclusion_partition = self._get_partition_id(conclusion_id)
        self.partition_graphs[conclusion_partition].add_rule(premise_ids, conclusion_id, confidence)
        
        # Also add to partitions containing premises
        premise_partitions = set(self._get_partition_id(p_id) for p_id in premise_ids)
        for partition_id in premise_partitions:
            if partition_id != conclusion_partition:
                self.partition_graphs[partition_id].add_rule(premise_ids, conclusion_id, confidence)
                
        return self
    
    def add_hierarchy(self, child_id: int, parent_id: int) -> 'DistributedKnowledgeGraph':
        """Add hierarchical relationship to appropriate partitions"""
        child_partition = self._get_partition_id(child_id)
        parent_partition = self._get_partition_id(parent_id)
        
        # Add to child's partition
        self.partition_graphs[child_partition].add_hierarchy(child_id, parent_id)
        
        # If parent is in different partition, add there too
        if child_partition != parent_partition:
            self.partition_graphs[parent_partition].add_hierarchy(child_id, parent_id)
            
        return self
    
    def reason(self, active_entities: List[int], max_hops: int = 3) -> Tuple[Set[int], Dict, Dict, Dict]:
        """
        Distributed reasoning across all partitions
        
        This executes reasoning on each partition and then merges the results
        """
        # Group active entities by partition
        partition_active_entities = defaultdict(list)
        for entity_id in active_entities:
            partition_id = self._get_partition_id(entity_id)
            partition_active_entities[partition_id].append(entity_id)
            
        # Execute reasoning on each relevant partition
        partition_results = []
        for partition_id, entities in partition_active_entities.items():
            if entities:  # Only process partitions with active entities
                result = self.partition_graphs[partition_id].reason(entities, max_hops)
                partition_results.append(result)
                
        # If no results, return empty
        if not partition_results:
            return set(), {}, {}, {}
            
        # Merge results from all partitions
        all_inferred = set()
        all_reasoning_steps = {}
        all_confidences = {}
        all_class_scores = defaultdict(float)
        
        for inferred, reasoning_steps, confidences, class_scores in partition_results:
            all_inferred.update(inferred)
            all_reasoning_steps.update(reasoning_steps)
            all_confidences.update(confidences)
            
            # Merge class scores, taking maximum confidence
            for class_id, score in class_scores.items():
                all_class_scores[class_id] = max(all_class_scores[class_id], score)
                
        return all_inferred, all_reasoning_steps, all_confidences, dict(all_class_scores)
    
    def save_to_disk(self) -> None:
        """Save the distributed knowledge graph to disk"""
        if not self.storage_dir:
            raise ValueError("No storage directory specified")
            
        # Save each partition
        for i, graph in enumerate(self.partition_graphs):
            graph.save_to_disk()
            
        # Save entity partition mapping
        with open(os.path.join(self.storage_dir, 'entity_partition_map.pkl'), 'wb') as f:
            pickle.dump(self.entity_partition_map, f)
            
        # Save metadata
        metadata = {
            'name': self.name,
            'num_partitions': self.num_partitions,
            'symbol_offset': self.symbol_offset,
            'num_classes': self.num_classes
        }
        
        with open(os.path.join(self.storage_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load_from_disk(cls, directory: str) -> 'DistributedKnowledgeGraph':
        """Load distributed graph from disk"""
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
            
        # Load metadata
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        # Create new instance
        graph = cls(
            name=metadata['name'],
            storage_dir=directory,
            num_partitions=metadata['num_partitions']
        )
        
        graph.symbol_offset = metadata['symbol_offset']
        graph.num_classes = metadata['num_classes']
        
        # Load entity partition mapping
        with open(os.path.join(directory, 'entity_partition_map.pkl'), 'rb') as f:
            graph.entity_partition_map = pickle.load(f)
            
        # Load each partition
        graph.partition_graphs = []
        for i in range(graph.num_partitions):
            partition_dir = os.path.join(directory, f"partition_{i}")
            if os.path.exists(partition_dir):
                partition_graph = SparseKnowledgeGraph.load_from_disk(partition_dir)
                graph.partition_graphs.append(partition_graph)
            else:
                # Create empty partition if not found
                graph.partition_graphs.append(
                    SparseKnowledgeGraph(
                        name=f"{graph.name}_partition_{i}",
                        storage_dir=partition_dir
                    )
                )
                
        return graph
    
    def get_entity(self, entity_id: int) -> Tuple[Optional[str], Optional[Dict]]:
        """Get entity name and attributes"""
        if entity_id in self.entity_partition_map:
            partition_id = self.entity_partition_map[entity_id]
            if entity_id in self.partition_graphs[partition_id].entities:
                name = self.partition_graphs[partition_id].entities[entity_id]
                attrs = self.partition_graphs[partition_id].entity_attrs.get(entity_id, {})
                return name, attrs
        return None, None