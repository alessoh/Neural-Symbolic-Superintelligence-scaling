"""
Run the Medical NEXUS model on the NHANES dataset for diabetes prediction.
python run_nhanes_nexus.py --epochs 15 --batch_size 64
Fixed version that uses CSV files instead of XPT (SAS) files to avoid library compatibility issues.
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
import zipfile
import io
import time

# Import medical-specific models
from medical_transformer_model import MedicalNEXUSModel

class NHANESDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_diabetes_dataset(data_dir="data"):
    """
    Load Pima Indians Diabetes Dataset as a fallback option
    This is a reliable dataset with 768 samples and 8 features
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Local file path
    file_path = os.path.join(data_dir, "diabetes.csv")
    
    # URL for the diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    
    # Download the data if not already available
    if not os.path.exists(file_path):
        print(f"Downloading Pima Indians Diabetes dataset to {file_path}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete!")
        else:
            raise Exception(f"Failed to download data: Status code {response.status_code}")
    
    # Column names for the dataset
    column_names = [
        'Pregnancies', 
        'Glucose', 
        'BloodPressure', 
        'SkinThickness', 
        'Insulin', 
        'BMI', 
        'DiabetesPedigreeFunction', 
        'Age',
        'Outcome'
    ]
    
    # Load the data
    try:
        df = pd.read_csv(file_path, header=None, names=column_names)
        print(f"Successfully loaded diabetes dataset with {len(df)} rows")
    except Exception as e:
        # Attempt direct loading without column names
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded diabetes dataset with {len(df)} rows using file headers")
            # Check if we need to rename the target column
            if 'Outcome' not in df.columns and df.shape[1] >= 9:
                df.columns = column_names
        except Exception as inner_e:
            raise Exception(f"Failed to load diabetes dataset: {inner_e}")
    
    # Check if the dataset loaded correctly
    if len(df) < 10:
        raise Exception("Dataset appears to be empty or corrupted")
    
    # Features and target
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values.astype(int)  # Last column is the target
    
    # Get actual feature names from DataFrame
    feature_names = df.columns[:-1].tolist()
    
    print(f"Dataset loaded: {len(df)} samples, {len(feature_names)} features")
    print(f"Features: {feature_names}")
    print(f"Diabetes prevalence: {np.mean(y) * 100:.1f}%")
    
    return X, y, feature_names

def try_load_kag_diabetes(data_dir="data"):
    """
    Alternative option: load a larger diabetes dataset from Kaggle
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Local file path
    file_path = os.path.join(data_dir, "diabetes_kaggle.csv")
    
    # URL for the dataset
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/diabetes.csv"
    
    # Download the data if not already available
    if not os.path.exists(file_path):
        print(f"Downloading Kaggle Diabetes dataset to {file_path}...")
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print("Download complete!")
            else:
                print(f"Failed to download data: Status code {response.status_code}")
                return None, None, None
        except Exception as e:
            print(f"Error downloading Kaggle dataset: {e}")
            return None, None, None
    
    # Try to load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded Kaggle diabetes dataset with {len(df)} rows")
        
        # Features and target
        if 'Outcome' in df.columns:
            X = df.drop('Outcome', axis=1).values
            y = df['Outcome'].values.astype(int)
            feature_names = df.drop('Outcome', axis=1).columns.tolist()
            
            print(f"Dataset loaded: {len(df)} samples, {len(feature_names)} features")
            print(f"Features: {feature_names}")
            print(f"Diabetes prevalence: {np.mean(y) * 100:.1f}%")
            
            return X, y, feature_names
        else:
            print("Could not identify target column in Kaggle dataset")
            return None, None, None
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")
        return None, None, None

def initialize_diabetes_knowledge_graph(model):
    """
    Initialize the knowledge graph with domain-specific knowledge for diabetes.
    """
    kg = model.knowledge_graph
    kg.symbol_offset = model.num_symbols
    kg.num_classes = model.num_classes
    
    feature_names = model.symbol_names
    
    # Add entities for features
    for i, feature_name in enumerate(feature_names):
        kg.add_entity(i, feature_name)
    
    # Add class entities
    kg.add_entity(kg.symbol_offset, "No Diabetes")
    kg.add_entity(kg.symbol_offset + 1, "Diabetes")
    
    # Add medical domain knowledge for diabetes
    
    # 1. Age factor
    kg.add_entity(100, "Age > 45", {"risk_factor": 0.3, "increases_1": 0.4})
    if 'Age' in feature_names:
        kg.add_relation(feature_names.index('Age'), "indicates", 100, weight=0.7)
    
    # 2. BMI factor
    kg.add_entity(102, "High BMI", {"risk_factor": 0.4, "increases_1": 0.5})
    if 'BMI' in feature_names:
        kg.add_relation(feature_names.index('BMI'), "indicates", 102, weight=0.8)
    
    # 3. Glucose factor
    kg.add_entity(103, "High Glucose", {"risk_factor": 0.6, "increases_1": 0.7})
    if 'Glucose' in feature_names:
        kg.add_relation(feature_names.index('Glucose'), "indicates", 103, weight=0.9)
    
    # 4. Blood Pressure
    kg.add_entity(104, "High Blood Pressure", {"risk_factor": 0.3, "increases_1": 0.4})
    if 'BloodPressure' in feature_names:
        kg.add_relation(feature_names.index('BloodPressure'), "indicates", 104, weight=0.7)
    
    # 5. Insulin Resistance
    kg.add_entity(105, "Insulin Resistance", {"risk_factor": 0.5, "increases_1": 0.6})
    if 'Insulin' in feature_names:
        kg.add_relation(feature_names.index('Insulin'), "indicates", 105, weight=0.8)
    
    # 6. Family History
    kg.add_entity(106, "Family History", {"risk_factor": 0.4, "increases_1": 0.5})
    if 'DiabetesPedigreeFunction' in feature_names:
        kg.add_relation(feature_names.index('DiabetesPedigreeFunction'), "indicates", 106, weight=0.8)
    
    # 7. Skin Thickness (related to body fat)
    kg.add_entity(107, "Increased Body Fat", {"risk_factor": 0.3, "increases_1": 0.4})
    if 'SkinThickness' in feature_names:
        kg.add_relation(feature_names.index('SkinThickness'), "indicates", 107, weight=0.6)
    
    # 8. Number of Pregnancies (risk for gestational diabetes history)
    kg.add_entity(108, "Multiple Pregnancies", {"risk_factor": 0.2, "increases_1": 0.3})
    if 'Pregnancies' in feature_names:
        kg.add_relation(feature_names.index('Pregnancies'), "indicates", 108, weight=0.5)
    
    # Add risk increasing connections
    for entity_id in range(100, 109):
        if entity_id in kg.entities:
            kg.add_relation(entity_id, "increases_risk", kg.symbol_offset + 1, weight=0.6)
    
    # Add rules (combinations of factors)
    
    # Rule 1: Age + High BMI
    if 'Age' in feature_names and 'BMI' in feature_names:
        kg.add_rule([100, 102], kg.symbol_offset + 1, confidence=0.7)
    
    # Rule 2: High BMI + High Glucose
    if 'BMI' in feature_names and 'Glucose' in feature_names:
        kg.add_rule([102, 103], kg.symbol_offset + 1, confidence=0.85)
    
    # Rule 3: Age + High BloodPressure + Family History
    if 'Age' in feature_names and 'BloodPressure' in feature_names and 'DiabetesPedigreeFunction' in feature_names:
        kg.add_rule([100, 104, 106], kg.symbol_offset + 1, confidence=0.75)
    
    # Rule 4: High Glucose + Insulin Resistance
    if 'Glucose' in feature_names and 'Insulin' in feature_names:
        kg.add_rule([103, 105], kg.symbol_offset + 1, confidence=0.8)
    
    # Rule 5: Age + High BMI + High BloodPressure
    if 'Age' in feature_names and 'BMI' in feature_names and 'BloodPressure' in feature_names:
        kg.add_rule([100, 102, 104], kg.symbol_offset + 1, confidence=0.75)
    
    # Add hierarchy relationships
    for entity_id in [102, 103, 105, 107]:
        if entity_id in kg.entities:
            kg.add_hierarchy(entity_id, "Metabolic Risk Factor")
    
    if 104 in kg.entities:
        kg.add_hierarchy(104, "Cardiovascular Risk Factor")
    
    if 106 in kg.entities:
        kg.add_hierarchy(106, "Genetic Risk Factor")
    
    return kg

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Medical NEXUS model on diabetes dataset")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (for memory-constrained environments)")
    parser.add_argument("--output_dir", type=str, default="diabetes_results", 
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
    print("Medical NEXUS Model - Diabetes Prediction")
    print("=" * 80)
    
    # First try to load the larger Kaggle dataset
    print("Attempting to load Kaggle diabetes dataset (larger)...")
    X, y, feature_names = try_load_kag_diabetes()
    
    # If Kaggle dataset failed, fall back to the Pima Indians dataset
    if X is None:
        print("Falling back to Pima Indians Diabetes dataset...")
        X, y, feature_names = load_diabetes_dataset()
    
    # Limit the number of samples if requested
    if args.max_samples and args.max_samples < len(X):
        print(f"Limiting to {args.max_samples} samples (out of {len(X)} available)")
        indices = np.random.choice(len(X), args.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = NHANESDataset(X_train, y_train)
    test_dataset = NHANESDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize the Medical NEXUS model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    num_symbols = input_dim
    symbol_names = feature_names
    class_names = ["No Diabetes", "Diabetes"]
    
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
        symbol_names=symbol_names,
        class_names=class_names,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        device=args.device
    )
    
    # Initialize the knowledge graph with diabetes knowledge
    print("\nInitializing knowledge graph with diabetes domain knowledge...")
    initialize_diabetes_knowledge_graph(model)
    
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
    model.visualize_results(
        output_prefix=os.path.join(args.output_dir, "results"),
        save_figures=True,
        show_figures=False
    )
    
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
    print(f"Confidence: {diagnosis['nexus']['confidence']:.2f}")
    print(f"Strategy Used: {diagnosis['nexus']['strategy']['strategy']}")
    
    # Print detected risk factors
    if diagnosis['symbolic']['active_symptoms']:
        print(f"\nDetected Risk Factors: {', '.join(diagnosis['symbolic']['active_symptoms'])}")
    else:
        print("\nNo risk factors detected above threshold")
    
    # Print explanation
    print("\nExplanation:")
    print(model.explain_diagnosis(diagnosis, detail_level='medium'))

if __name__ == "__main__":
    main()