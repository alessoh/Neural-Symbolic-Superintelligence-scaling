"""
Run the Medical NEXUS model on a large medical dataset.
This script uses the Kaggle Heart Failure Prediction dataset and generates 
synthetic data expansion to create a dataset with more than 5,000 samples.

Usage:
python run_large_medical_nexus_fixed.py --epochs 15 --batch_size 64
"""

import torch
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import requests
import time
from sklearn.utils import resample

# Import the medical models
from medical_transformer_model import MedicalNEXUSModel

# Custom dataset class
class MedicalDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def download_heart_failure_dataset(data_dir="data"):
    """
    Download the Heart Failure Prediction dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Local file path
    file_path = os.path.join(data_dir, "heart_failure.csv")
    
    # If file already exists, just return the path
    if os.path.exists(file_path):
        print(f"Heart failure dataset already exists at {file_path}")
        return file_path
    
    # URL for the heart failure dataset
    # This is a reliable source for the Heart Failure Clinical Records Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart_failure_clinical_records_dataset.csv"
    
    print(f"Downloading heart failure dataset to {file_path}...")
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete!")
            return file_path
        else:
            # Try alternative source
            print(f"Failed to download from primary source. Status code: {response.status_code}")
            url2 = "https://storage.googleapis.com/kaggle-data-sets/435/13938/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210813%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210813T125841Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=8b14dba495037a933324a4a13c453f70866d528e1fd7ce51b401e7283f72d99b42740875c64cddae598b42bac6f5eb084be7b566fbb6fbda9648621eaaeb71b7a7df67fcfd11f6d3f0af8a5c72c7dc36f88e165de7b0c42d9f2b8219a64eb24b0b55d53866b31fc9fdf35a2ad95d224bc7d2e246e1866e4582e3cca7d1b9cac89b16e2587f5c1c974b40d83c8085b6b6a62f3f591f07cc66ab97d99cb851353b8164ca1f16651db5a7be70940b65c0eafa02bfe462238984746a8082b9e7e1ae9fe8d6bb1eff14be828e4c6fc47a739fc0c0a0ce8c0fd8f991877e8ad7fb0a6a66add8c6a83add87c3f5dee1a0bb0ee3489ebe2bcaa9d4ed45dd9a5ae"
            try:
                response = requests.get(url2)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print("Download from alternative source complete!")
                    return file_path
                else:
                    print(f"Failed to download from alternative source. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading from alternative source: {e}")
            
            # If both sources fail, create a synthetic dataset
            print("Creating synthetic heart failure dataset...")
            create_synthetic_heart_failure_dataset(file_path)
            return file_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Creating synthetic heart failure dataset...")
        create_synthetic_heart_failure_dataset(file_path)
        return file_path

def create_synthetic_heart_failure_dataset(file_path):
    """
    Create a synthetic heart failure dataset if download fails
    """
    # Create a synthetic dataset with similar properties to the heart failure dataset
    # Define column names and sample data
    columns = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
        'ejection_fraction', 'high_blood_pressure', 'platelets', 
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'
    ]
    
    # Generate 300 synthetic records as a starting point
    np.random.seed(42)  # for reproducibility
    
    data = {
        'age': np.random.normal(60, 10, 300).clip(20, 95),
        'anaemia': np.random.choice([0, 1], 300),
        'creatinine_phosphokinase': np.random.lognormal(5, 1, 300).clip(20, 7000),
        'diabetes': np.random.choice([0, 1], 300),
        'ejection_fraction': np.random.normal(38, 12, 300).clip(14, 80).astype(int),
        'high_blood_pressure': np.random.choice([0, 1], 300),
        'platelets': np.random.normal(260000, 70000, 300).clip(50000, 850000),
        'serum_creatinine': np.random.lognormal(0, 0.5, 300).clip(0.5, 9.5),
        'serum_sodium': np.random.normal(137, 4, 300).clip(113, 150).astype(int),
        'sex': np.random.choice([0, 1], 300),
        'smoking': np.random.choice([0, 1], 300),
        'time': np.random.randint(4, 300, 300),
    }
    
    # Generate outcome with some realistic relationships
    # Higher age, lower ejection fraction, and higher serum creatinine increase death probability
    death_prob = (
        0.3 + 
        0.01 * (data['age'] - 50) / 45 + 
        0.02 * (50 - data['ejection_fraction']) / 30 + 
        0.3 * (data['serum_creatinine'] - 0.5) / 9 +
        0.1 * data['high_blood_pressure'] +
        0.05 * data['diabetes']
    ).clip(0, 0.9)
    
    data['DEATH_EVENT'] = np.random.binomial(1, death_prob)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    df.to_csv(file_path, index=False)
    print(f"Synthetic dataset with {len(df)} rows created and saved to {file_path}")
    return df

def expand_dataset_to_5000(df, target_size=5000):
    """
    Expand the dataset to have at least 5000 samples using resampling and noise addition
    """
    original_size = len(df)
    print(f"Expanding dataset from {original_size} to {target_size} samples...")
    
    # How many copies we need
    copies_needed = int(np.ceil(target_size / original_size))
    
    # Create expanded dataframe
    expanded_data = []
    for i in range(copies_needed):
        # For the first copy, use the original data
        if i == 0:
            expanded_data.append(df.copy())
        else:
            # For subsequent copies, resample with noise
            sample = df.copy()
            
            # Add noise to continuous variables
            for col in ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                       'platelets', 'serum_creatinine', 'serum_sodium', 'time']:
                if col in sample.columns:
                    # Add noise scaled to the standard deviation of the column
                    noise_scale = sample[col].std() * 0.1  # 10% of std as noise
                    sample[col] = sample[col] + np.random.normal(0, noise_scale, len(sample))
                    
                    # Ensure values stay in reasonable ranges
                    if col == 'age':
                        sample[col] = sample[col].clip(20, 95)
                    elif col == 'ejection_fraction':
                        sample[col] = sample[col].clip(14, 80).round().astype(int)
                    elif col == 'serum_sodium':
                        sample[col] = sample[col].clip(113, 150).round().astype(int)
                    elif col == 'creatinine_phosphokinase':
                        sample[col] = sample[col].clip(20, 7000)
                    elif col == 'platelets':
                        sample[col] = sample[col].clip(50000, 850000)
                    elif col == 'serum_creatinine':
                        sample[col] = sample[col].clip(0.5, 9.5)
                    elif col == 'time':
                        sample[col] = sample[col].clip(4, 300).round().astype(int)
            
            # Occasionally flip binary variables
            for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']:
                if col in sample.columns:
                    # 5% chance of flipping each binary value
                    flip_mask = np.random.random(len(sample)) < 0.05
                    sample.loc[flip_mask, col] = 1 - sample.loc[flip_mask, col]
            
            # Recalculate death event with some randomness
            # This helps maintain the original relationship patterns with some variance
            if 'DEATH_EVENT' in sample.columns:
                death_prob = (
                    0.3 + 
                    0.01 * (sample['age'] - 50) / 45 + 
                    0.02 * (50 - sample['ejection_fraction']) / 30 + 
                    0.3 * (sample['serum_creatinine'] - 0.5) / 9 +
                    0.1 * sample['high_blood_pressure'] +
                    0.05 * sample['diabetes']
                ).clip(0, 0.9)
                
                # Add some randomness to the death probability
                death_prob = (death_prob + np.random.normal(0, 0.1, len(sample))).clip(0, 1)
                sample['DEATH_EVENT'] = np.random.binomial(1, death_prob)
            
            expanded_data.append(sample)
    
    # Combine all the expanded data
    expanded_df = pd.concat(expanded_data, ignore_index=True)
    
    # Trim to exact target size
    if len(expanded_df) > target_size:
        expanded_df = expanded_df.iloc[:target_size]
    
    print(f"Dataset expanded to {len(expanded_df)} samples")
    return expanded_df

def initialize_heart_failure_knowledge_graph(model):
    """
    Initialize the knowledge graph with domain-specific knowledge for heart failure.
    """
    kg = model.knowledge_graph
    kg.symbol_offset = model.num_symbols
    kg.num_classes = model.num_classes
    
    feature_names = model.symbol_names
    
    # Add entities for features
    for i, feature_name in enumerate(feature_names):
        kg.add_entity(i, feature_name)
    
    # Add class entities
    kg.add_entity(kg.symbol_offset, "Heart Failure Death")
    kg.add_entity(kg.symbol_offset + 1, "Heart Failure Survival")
    
    # Create a function to safely find feature indices
    def find_feature_index(name):
        for i, feature in enumerate(feature_names):
            if name == feature:
                return i
        return None
    
    # Create indexes to easily access feature groups
    age_idx = find_feature_index('age')
    anaemia_idx = find_feature_index('anaemia')
    cpk_idx = find_feature_index('creatinine_phosphokinase')
    diabetes_idx = find_feature_index('diabetes')
    ejection_fraction_idx = find_feature_index('ejection_fraction')
    high_bp_idx = find_feature_index('high_blood_pressure')
    platelets_idx = find_feature_index('platelets')
    serum_creatinine_idx = find_feature_index('serum_creatinine')
    serum_sodium_idx = find_feature_index('serum_sodium')
    sex_idx = find_feature_index('sex')
    smoking_idx = find_feature_index('smoking')
    time_idx = find_feature_index('time')
    
    # Add medical domain knowledge entities
    
    # 1. Advanced age (> 65)
    kg.add_entity(100, "Advanced Age", {"risk_factor": 0.5, "increases_0": 0.6})
    if age_idx is not None:
        kg.add_relation(age_idx, "indicates", 100, weight=0.7)
    
    # 2. Low ejection fraction (< 30%)
    kg.add_entity(101, "Low Ejection Fraction", {"risk_factor": 0.7, "increases_0": 0.8})
    if ejection_fraction_idx is not None:
        kg.add_relation(ejection_fraction_idx, "indicates", 101, weight=0.85)
    
    # 3. High serum creatinine (> 1.5 mg/dL)
    kg.add_entity(102, "High Serum Creatinine", {"risk_factor": 0.6, "increases_0": 0.7})
    if serum_creatinine_idx is not None:
        kg.add_relation(serum_creatinine_idx, "indicates", 102, weight=0.8)
    
    # 4. Low serum sodium (< 135 mEq/L)
    kg.add_entity(103, "Low Serum Sodium", {"risk_factor": 0.5, "increases_0": 0.6})
    if serum_sodium_idx is not None:
        kg.add_relation(serum_sodium_idx, "indicates", 103, weight=0.7)
    
    # 5. High CPK (> 800 units/L)
    kg.add_entity(104, "High CPK", {"risk_factor": 0.4, "increases_0": 0.5})
    if cpk_idx is not None:
        kg.add_relation(cpk_idx, "indicates", 104, weight=0.6)
    
    # 6. Hypertension
    kg.add_entity(105, "Hypertension", {"risk_factor": 0.4, "increases_0": 0.5})
    if high_bp_idx is not None:
        kg.add_relation(high_bp_idx, "indicates", 105, weight=0.7)
    
    # 7. Diabetes
    kg.add_entity(106, "Diabetes", {"risk_factor": 0.4, "increases_0": 0.5})
    if diabetes_idx is not None:
        kg.add_relation(diabetes_idx, "indicates", 106, weight=0.6)
    
    # 8. Anaemia
    kg.add_entity(107, "Anaemia", {"risk_factor": 0.3, "increases_0": 0.4})
    if anaemia_idx is not None:
        kg.add_relation(anaemia_idx, "indicates", 107, weight=0.6)
    
    # 9. Smoking
    kg.add_entity(108, "Smoking", {"risk_factor": 0.4, "increases_0": 0.5})
    if smoking_idx is not None:
        kg.add_relation(smoking_idx, "indicates", 108, weight=0.65)
    
    # 10. Follow-up time (shorter follow-up with death suggests more severe condition)
    kg.add_entity(109, "Short Follow-up", {"risk_factor": 0.3, "increases_0": 0.4})
    if time_idx is not None:
        kg.add_relation(time_idx, "indicates", 109, weight=0.5)
    
    # Add risk increasing connections to heart failure death
    for entity_id in range(100, 110):
        if entity_id in kg.entities:
            kg.add_relation(entity_id, "increases_risk", kg.symbol_offset, weight=0.7)
    
    # Add rules (combinations of factors)
    
    # Rule 1: Advanced age + Low ejection fraction
    if age_idx is not None and ejection_fraction_idx is not None:
        kg.add_rule([100, 101], kg.symbol_offset, confidence=0.8)
    
    # Rule 2: Low ejection fraction + High serum creatinine
    if ejection_fraction_idx is not None and serum_creatinine_idx is not None:
        kg.add_rule([101, 102], kg.symbol_offset, confidence=0.85)
    
    # Rule 3: Advanced age + Hypertension + Diabetes
    if age_idx is not None and high_bp_idx is not None and diabetes_idx is not None:
        kg.add_rule([100, 105, 106], kg.symbol_offset, confidence=0.75)
    
    # Rule 4: Low ejection fraction + Low serum sodium
    if ejection_fraction_idx is not None and serum_sodium_idx is not None:
        kg.add_rule([101, 103], kg.symbol_offset, confidence=0.8)
    
    # Rule 5: High serum creatinine + Anaemia
    if serum_creatinine_idx is not None and anaemia_idx is not None:
        kg.add_rule([102, 107], kg.symbol_offset, confidence=0.75)
    
    # Rule 6: Smoking + Hypertension + Diabetes (multiple risk factors)
    if smoking_idx is not None and high_bp_idx is not None and diabetes_idx is not None:
        kg.add_rule([108, 105, 106], kg.symbol_offset, confidence=0.8)
    
    # Add hierarchy relationships
    kg.add_hierarchy(101, "Cardiac Function Marker")
    kg.add_hierarchy(104, "Cardiac Function Marker")
    
    kg.add_hierarchy(102, "Kidney Function Marker")
    kg.add_hierarchy(103, "Electrolyte Balance Marker")
    
    kg.add_hierarchy(105, "Cardiovascular Risk Factor")
    kg.add_hierarchy(106, "Metabolic Risk Factor")
    kg.add_hierarchy(107, "Hematologic Risk Factor")
    kg.add_hierarchy(108, "Lifestyle Risk Factor")
    
    return kg

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Medical NEXUS model on expanded heart failure dataset")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument("--target_size", type=int, default=5000,
                        help="Target size of the expanded dataset")
    parser.add_argument("--output_dir", type=str, default="heart_failure_results", 
                        help="Output directory")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dimension for the neural model")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer layers")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Medical NEXUS Model - Expanded Heart Failure Dataset")
    print("=" * 80)
    
    # Download the heart failure dataset
    csv_path = download_heart_failure_dataset()
    if not csv_path:
        print("Failed to retrieve the dataset. Exiting.")
        return
        
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns)}")
        
        if 'DEATH_EVENT' in df.columns:
            print(f"Death events: {df['DEATH_EVENT'].sum()} ({df['DEATH_EVENT'].mean()*100:.1f}%)")
        
        # Expand the dataset to at least 5000 samples
        if len(df) < args.target_size:
            df = expand_dataset_to_5000(df, args.target_size)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Prepare features and target
    X = df.drop('DEATH_EVENT', axis=1).values
    y = df['DEATH_EVENT'].values
    feature_names = list(df.drop('DEATH_EVENT', axis=1).columns)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = MedicalDataset(X_train, y_train)
    test_dataset = MedicalDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize the Medical NEXUS model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    num_symbols = input_dim
    class_names = ["Heart Failure Death", "Heart Failure Survival"]
    
    print(f"\nCreating Medical NEXUS model with:")
    print(f"- Input dimensions: {input_dim}")
    print(f"- Number of features: {len(feature_names)}")
    print(f"- Features: {', '.join(feature_names)}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Classes: {', '.join(class_names)}")
    print(f"- Training set size: {len(train_dataset)}")
    print(f"- Test set size: {len(test_dataset)}")
    print(f"- Embedding dimension: {args.embed_dim}")
    print(f"- Number of transformer layers: {args.num_layers}")
    
    # Create the model
    model = MedicalNEXUSModel(
        input_dim=input_dim,
        num_classes=num_classes,
        num_symbols=num_symbols,
        symbol_names=feature_names,
        class_names=class_names,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        device=args.device
    )
    
    # Initialize the knowledge graph with heart failure domain knowledge
    print("\nInitializing knowledge graph with heart failure domain knowledge...")
    initialize_heart_failure_knowledge_graph(model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    print("\nTraining Medical NEXUS model...")
    start_time = time.time()
    model.train_neural(
        train_loader,
        num_epochs=args.epochs,
        lr=args.learning_rate,
        scheduler='cosine'
    )
    train_time = time.time() - start_time
    
    # Evaluate the model
    print("\nEvaluating model...")
    with torch.no_grad():
        results = model.evaluate(test_loader)
    
    # Save the model
    model.save_model(os.path.join(args.output_dir, "model"))
    
    # Export evaluation results
    model.export_results(os.path.join(args.output_dir, "evaluation_results.csv"))
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    try:
        model.visualize_results(
            output_prefix=os.path.join(args.output_dir, "results"),
            save_figures=True,
            show_figures=False
        )
    except Exception as e:
        print(f"Warning: Error generating visualizations: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Training completed in {train_time / 60:.2f} minutes")
    print("=" * 80)
    
    print("\nPerformance Summary:")
    print("-" * 40)
    print(f"Neural Model Accuracy: {results['neural']['accuracy']*100:.2f}%")
    print(f"Symbolic Model Accuracy: {results['symbolic']['accuracy']*100:.2f}%")
    print(f"NEXUS Model Accuracy: {results['nexus']['accuracy']*100:.2f}%")
    
    # Calculate improvement
    best_component = max(results['neural']['accuracy'], results['symbolic']['accuracy'])
    improvement = (results['nexus']['accuracy'] - best_component) * 100
    
    print(f"\nNEXUS improvement over best component: {improvement:.2f}%")
    
    # Print agreement analysis
    agreement = results['agreement_cases']
    total = sum(agreement.values())
    
    if total > 0:
        print("\nAgreement Analysis:")
        print(f"All models correct: {agreement['all_correct']} cases ({100*agreement['all_correct']/total:.1f}%)")
        print(f"Neural only correct: {agreement['neural_only']} cases ({100*agreement['neural_only']/total:.1f}%)")
        print(f"Symbolic only correct: {agreement['symbolic_only']} cases ({100*agreement['symbolic_only']/total:.1f}%)")
        print(f"NEXUS better than components: {agreement['nexus_better']} cases ({100*agreement['nexus_better']/total:.1f}%)")
        print(f"All models wrong: {agreement['all_wrong']} cases ({100*agreement['all_wrong']/total:.1f}%)")
    
    # Print metacognitive strategy usage
    strategy_stats = model.metacognitive.get_strategy_stats()
    
    print("\nMetacognitive Strategy Usage:")
    if 'neural' in strategy_stats:
        print(f"Neural strategy: {strategy_stats['neural']*100:.1f}%")
    if 'symbolic' in strategy_stats:
        print(f"Symbolic strategy: {strategy_stats['symbolic']*100:.1f}%")
    if 'hybrid' in strategy_stats:
        print(f"Hybrid strategy: {strategy_stats['hybrid']*100:.1f}%")
    
    # Run an example diagnosis
    print("\nExample Diagnosis:")
    print("-" * 40)
    
    # Use the first test sample
    sample_input = X_test[0:1]
    sample_tensor = torch.tensor(sample_input, dtype=torch.float32)
    
    # Generate diagnosis
    diagnosis = model.diagnose(sample_tensor, risk_level='medium')
    
    # Print diagnosis details
    print(f"Predicted Class: {diagnosis['nexus']['class_name']}")
    print(f"True Class: {class_names[y_test[0]]}")
    print(f"Confidence: {diagnosis['nexus']['confidence']:.2f}")
    print(f"Strategy Used: {diagnosis['nexus']['strategy']['strategy']}")
    
    # Print detected features
    if diagnosis['symbolic']['active_symptoms']:
        detected = diagnosis['symbolic']['active_symptoms']
        print(f"\nDetected Significant Features: {', '.join(detected)}")
    else:
        print("\nNo significant features detected above threshold")
    
    # Print explanation
    print("\nExplanation:")
    print(model.explain_diagnosis(diagnosis, detail_level='medium'))

if __name__ == "__main__":
    main()