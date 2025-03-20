"""
Run the Medical NEXUS model on the UCI Heart Disease dataset.

This script uses the custom MedicalTransformerModel and MedicalNEXUSModel
instead of patching the existing classes, for a more robust solution.
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

# Import our medical-specific models
from medical_transformer_model import MedicalNEXUSModel, initialize_heart_disease_knowledge_graph

# Define a class for the UCI Heart Disease dataset
class HeartDiseaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_heart_disease_data(data_dir="data"):
    """
    Load the UCI Heart Disease dataset
    """
    # URL for the UCI Heart Disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Local file path
    file_path = os.path.join(data_dir, "heart_disease.csv")
    
    # Download the data if not already available
    if not os.path.exists(file_path):
        print(f"Downloading UCI Heart Disease dataset to {file_path}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete!")
        else:
            raise Exception(f"Failed to download data: Status code {response.status_code}")
    
    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load the data
    try:
        df = pd.read_csv(file_path, header=None, names=column_names)
    except:
        # Try with different parsing for missing values
        df = pd.read_csv(file_path, header=None, names=column_names, na_values='?')
    
    # Handle missing values - replace with median for numeric columns
    for col in df.columns:
        if df[col].dtype != object:  # Numeric columns
            df[col] = df[col].fillna(df[col].median())
    
    # Convert any remaining '?' to NaN and then to median values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Convert target to binary (0 = no disease, 1 = disease)
    y = (y > 0).astype(int)
    
    return X, y, column_names[:-1]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Medical NEXUS model on UCI Heart Disease dataset")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default="heart_disease_results", 
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
    print("Medical NEXUS Model - UCI Heart Disease Dataset")
    print("=" * 80)
    
    # Load the heart disease dataset
    print("Loading UCI Heart Disease dataset...")
    X, y, feature_names = load_heart_disease_data()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    test_dataset = HeartDiseaseDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize the Medical NEXUS model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    num_symbols = input_dim
    symbol_names = feature_names
    class_names = ["No Heart Disease", "Heart Disease"]
    
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
    
    # Initialize the knowledge graph with medical knowledge
    print("\nInitializing knowledge graph with medical domain knowledge...")
    initialize_heart_disease_knowledge_graph(model)
    
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
    print(f"NEXUS Model Accuracy: {results['neural']['accuracy']*100:.2f}%")
    
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
    
    # Print medical-specific metrics if available
    try:
        medical_summary = model.visualize_results()['medical']
        
        print("\nMedical-Specific Metrics:")
        
        # Risk level performance
        if 'risk_performance' in medical_summary:
            print("\nPerformance by Risk Level:")
            for risk, stats in medical_summary['risk_performance'].items():
                print(f"  {risk.capitalize()} Risk ({stats['count']} patients):")
                print(f"    - Neural: {stats['neural_accuracy']:.1f}%")
                print(f"    - Symbolic: {stats['symbolic_accuracy']:.1f}%")
                print(f"    - NEXUS: {stats['nexus_accuracy']:.1f}%")
        
        # Critical symptoms impact
        if 'critical_symptoms_impact' in medical_summary:
            print("\nImpact of Critical Symptoms:")
            with_critical = medical_summary['critical_symptoms_impact']['with_critical']
            without_critical = medical_summary['critical_symptoms_impact']['without_critical']
            
            print(f"  With Critical Symptoms ({with_critical['count']} patients):")
            print(f"    - Neural: {with_critical['neural_accuracy']:.1f}%")
            print(f"    - Symbolic: {with_critical['symbolic_accuracy']:.1f}%")
            print(f"    - NEXUS: {with_critical['nexus_accuracy']:.1f}%")
            
            print(f"  Without Critical Symptoms ({without_critical['count']} patients):")
            print(f"    - Neural: {without_critical['neural_accuracy']:.1f}%")
            print(f"    - Symbolic: {without_critical['symbolic_accuracy']:.1f}%")
            print(f"    - NEXUS: {without_critical['nexus_accuracy']:.1f}%")
        
        # Strategy by risk level
        if 'strategy_by_risk' in medical_summary:
            print("\nStrategy Selection by Risk Level:")
            for risk, strategies in medical_summary['strategy_by_risk'].items():
                print(f"  {risk.capitalize()} Risk:")
                for strategy, pct in strategies.items():
                    print(f"    - {strategy.capitalize()}: {pct:.1f}%")
    except Exception as e:
        print(f"\nNote: Could not display detailed medical metrics. Error: {str(e)}")
    
    print(f"\nAll results saved to: {args.output_dir}")
    
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
    
    # Print detected symptoms
    if diagnosis['symbolic']['active_symptoms']:
        print(f"\nDetected Symptoms: {', '.join(diagnosis['symbolic']['active_symptoms'])}")
    else:
        print("\nNo symptoms detected above threshold")
    
    # Print explanation
    print("\nExplanation:")
    print(model.explain_diagnosis(diagnosis, detail_level='medium'))

if __name__ == "__main__":
    main()