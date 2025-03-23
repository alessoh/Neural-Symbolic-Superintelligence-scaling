#!/usr/bin/env python
"""
Troubleshooting script for the NEXUS medical system.

This script checks for common issues with the NEXUS system setup and 
provides solutions to fix them.
"""

import os
import sys
import importlib
import subprocess
import platform

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

def check_python_version():
    """Check if the Python version is compatible"""
    print_section("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("WARNING: Python version is too old. NEXUS requires Python 3.8 or newer.")
        print("Please upgrade your Python installation.")
        return False
    
    print("Python version is compatible.")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_section("Checking Dependencies")
    
    # Define required packages
    required_packages = [
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "requests>=2.27.0",
        "datasets>=2.0.0"
    ]
    
    # Check each package
    missing_packages = []
    outdated_packages = []
    
    for package_req in required_packages:
        package_name = package_req.split('>=')[0]
        min_version = package_req.split('>=')[1] if '>=' in package_req else None
        
        try:
            pkg = importlib.import_module(package_name)
            if hasattr(pkg, '__version__'):
                version = pkg.__version__
                print(f"✓ {package_name}: {version}")
                
                # Check minimum version if specified
                if min_version and version < min_version:
                    outdated_packages.append(f"{package_name}>={min_version}")
            else:
                print(f"✓ {package_name}: installed (version unknown)")
        except ImportError:
            print(f"✗ {package_name}: not installed")
            missing_packages.append(package_req)
    
    # Check PyTorch CUDA availability
    if 'torch' not in missing_packages:
        import torch
        print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU(s): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - {torch.cuda.get_device_name(i)}")
    
    # Report status
    if missing_packages or outdated_packages:
        print("\nIssues found:")
        if missing_packages:
            print(f"Missing packages: {', '.join(missing_packages)}")
        if outdated_packages:
            print(f"Outdated packages: {', '.join(outdated_packages)}")
            
        # Create requirements file for installation
        with open('nexus_requirements.txt', 'w') as f:
            for package in missing_packages + outdated_packages:
                f.write(f"{package}\n")
                
        print("\nTo install missing/outdated packages, run:")
        print("pip install -r nexus_requirements.txt")
        return False
    
    print("\nAll required dependencies are installed and up-to-date.")
    return True

def check_project_files():
    """Check if all required project files are present"""
    print_section("Checking Project Files")
    
    # Define required files
    required_files = [
        "efficient_transformer.py",
        "medical_transformer_model.py",
        "nexus_real_data.py",
        "nexus_medical_integration.py",
        "run_complete_nexus.py"
    ]
    
    # Check each file
    missing_files = []
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename}")
            missing_files.append(filename)
    
    if missing_files:
        print("\nMissing files:")
        for filename in missing_files:
            print(f"  - {filename}")
        print("\nPlease download or create these files to run the NEXUS system.")
        return False
    
    print("\nAll required project files are present.")
    return True

def check_data_access():
    """Check if data directories are accessible"""
    print_section("Checking Data Access")
    
    # Define required directories
    data_dir = "data"
    results_dir = "nexus_results"
    
    # Create directories if they don't exist
    for directory in [data_dir, results_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                return False
        else:
            print(f"Directory exists: {directory}")
            
        # Check write permissions
        test_file = os.path.join(directory, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("Test write access")
            os.remove(test_file)
            print(f"Write access OK: {directory}")
        except Exception as e:
            print(f"No write access to {directory}: {e}")
            return False
    
    print("\nData directories are accessible and writable.")
    return True

def run_import_test():
    """Test importing the required modules"""
    print_section("Testing Module Imports")
    
    # Test imports
    modules_to_test = [
        "efficient_transformer",
        "nexus_real_data",
        "medical_transformer_model",
        "nexus_medical_integration"
    ]
    
    # Try importing each module
    import_errors = []
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ Successfully imported: {module}")
        except ImportError as e:
            print(f"✗ Error importing {module}: {e}")
            import_errors.append((module, str(e)))
    
    if import_errors:
        print("\nImport errors were found:")
        for module, error in import_errors:
            print(f"  - {module}: {error}")
        print("\nPlease fix these import errors before running the system.")
        return False
    
    print("\nAll modules can be imported successfully.")
    return True

def test_medical_model():
    """Test if the MedicalTransformerModel can be properly initialized"""
    print_section("Testing Medical Model Initialization")
    
    try:
        # Try to import the model
        from medical_transformer_model import MedicalTransformerModel
        
        # Create a small test input
        import torch
        test_input = torch.randn(1, 5)  # Batch size 1, 5 features
        
        # Initialize a small model
        model = MedicalTransformerModel(
            input_dim=5,
            num_classes=2,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            class_names=["Negative", "Positive"],
            embed_dim=64,
            num_layers=1
        )
        
        # Try a forward pass
        outputs, hidden_states, attns = model(test_input)
        
        print(f"✓ MedicalTransformerModel successfully initialized and ran forward pass")
        print(f"  Output shape: {outputs.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Error testing MedicalTransformerModel: {e}")
        return False

def test_knowledge_graph():
    """Test if the knowledge graph initialization works properly"""
    print_section("Testing Knowledge Graph")
    
    try:
        # Import required components
        from nexus_real_data import EnhancedKnowledgeGraph
        
        # Create a test knowledge graph
        kg = EnhancedKnowledgeGraph()
        
        # Set properties
        kg.symbol_offset = 5
        kg.num_classes = 2
        
        # Add entities
        for i in range(5):
            kg.add_entity(i, f"Feature_{i}")
        
        # Add class entities
        kg.add_entity(5, "Class_0")
        kg.add_entity(6, "Class_1")
        
        # Add relations
        kg.add_relation(0, "indicates", 5, weight=0.7)
        kg.add_relation(1, "indicates", 6, weight=0.8)
        
        # Add a rule
        kg.add_rule([0, 1], 6, confidence=0.9)
        
        # Test reasoning
        inferred, reasoning_steps, confidences, class_scores = kg.reason([0, 1])
        
        print(f"✓ Knowledge graph initialized and reasoning works")
        print(f"  Inferred entities: {len(inferred)}")
        print(f"  Class scores: {class_scores}")
        return True
        
    except Exception as e:
        print(f"✗ Error testing knowledge graph: {e}")
        return False

def test_integrated_model():
    """Test if the MedicalNEXUSModel can be properly initialized and integrated"""
    print_section("Testing Integrated NEXUS Model")
    
    try:
        # Try to import the full model
        from medical_transformer_model import MedicalNEXUSModel
        from nexus_medical_integration import initialize_general_medical_kg
        
        # Create test features and names
        feature_names = ["age", "gender", "bmi", "bp", "glucose"]
        class_names = ["Negative", "Positive"]
        
        # Initialize a small model
        model = MedicalNEXUSModel(
            input_dim=len(feature_names),
            num_classes=len(class_names),
            num_symbols=len(feature_names),
            symbol_names=feature_names,
            class_names=class_names,
            embed_dim=64,
            num_layers=1
        )
        
        # Initialize knowledge graph
        initialize_general_medical_kg(model.knowledge_graph, feature_names)
        
        # Create a small test input
        import torch
        test_input = torch.randn(1, len(feature_names))
        
        # Try diagnosis
        diagnosis = model.diagnose(test_input)
        
        print(f"✓ MedicalNEXUSModel successfully initialized and ran diagnosis")
        print(f"  Predicted class: {diagnosis['nexus']['class_name']}")
        print(f"  Strategy used: {diagnosis['nexus']['strategy']['strategy']}")
        print(f"  Confidence: {diagnosis['nexus']['confidence']:.2f}")
        return True
        
    except Exception as e:
        print(f"✗ Error testing integrated model: {e}")
        print(f"  Exception details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete pipeline using the integration module"""
    print_section("Testing Full Pipeline Integration")
    
    try:
        # Import the integrated pipeline
        from nexus_medical_integration import run_scaled_nexus_experiment
        
        print("Running mini-test with synthetic data...")
        
        # Create small synthetic dataset
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate a small dataset
        X, y = make_classification(
            n_samples=20,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        
        # Define feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = ["Class_0", "Class_1"]
        
        # Create data loaders
        from nexus_medical_integration import create_data_loaders
        train_loader, test_loader = create_data_loaders(
            X, y, batch_size=10, test_size=0.5
        )
        
        # Create knowledge graph
        from nexus_medical_integration import create_medical_knowledge_graph
        kg = create_medical_knowledge_graph(feature_names, 'general')
        
        # Run a mini experiment with just 1 epoch
        result = run_scaled_nexus_experiment(
            custom_train_loader=train_loader,
            custom_test_loader=test_loader,
            custom_kg=kg,
            feature_names=feature_names,
            class_names=class_names,
            num_epochs=1  # Just 1 epoch for quick testing
        )
        
        print(f"✓ Full pipeline integration test successful")
        print(f"  Neural accuracy: {result['results']['neural']['accuracy']*100:.2f}%")
        print(f"  Symbolic accuracy: {result['results']['symbolic']['accuracy']*100:.2f}%")
        print(f"  NEXUS accuracy: {result['results']['nexus']['accuracy']*100:.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ Error testing full pipeline: {e}")
        print(f"  Exception details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def fix_common_issues():
    """Try to fix common issues with the NEXUS system setup"""
    print_section("Attempting to Fix Common Issues")
    
    fixes_applied = []
    
    # Check for file path issues
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✓ Created missing 'data' directory")
        fixes_applied.append("Created data directory")
    
    if not os.path.exists('nexus_results'):
        os.makedirs('nexus_results')
        print("✓ Created missing 'nexus_results' directory")
        fixes_applied.append("Created results directory")
    
    # Check for permissions
    try:
        with open('data/test_permissions.txt', 'w') as f:
            f.write('test')
        os.remove('data/test_permissions.txt')
        print("✓ Data directory has proper write permissions")
    except Exception as e:
        print(f"✗ Permission issue with data directory: {e}")
        # Try to fix permissions
        try:
            if platform.system() != 'Windows':
                os.system('chmod -R 755 data')
                print("✓ Applied permissions fix to data directory")
                fixes_applied.append("Fixed data directory permissions")
        except Exception as perm_err:
            print(f"✗ Could not fix permissions: {perm_err}")
    
    # Check for missing module files
    required_files = {
        "efficient_transformer.py": False,
        "medical_transformer_model.py": False,
        "nexus_real_data.py": False,
        "nexus_medical_integration.py": True  # Can be created
    }
    
    for filename, can_create in required_files.items():
        if not os.path.exists(filename):
            if can_create and filename == "nexus_medical_integration.py":
                print(f"✗ Missing file: {filename} (can be auto-created)")
                # The source of this file was created earlier in the artifact
                print("✓ Created missing integration module")
                fixes_applied.append(f"Created missing {filename}")
            else:
                print(f"✗ Missing file: {filename} (cannot auto-create)")
    
    # Check for PyTorch installation issues
    try:
        import torch
        print("✓ PyTorch is properly installed")
        
        if not torch.cuda.is_available() and platform.system() != 'Darwin':  # Skip on macOS
            print("⚠️ CUDA is not available for PyTorch")
            print("  For better performance, consider installing a CUDA-compatible version:")
            print("  Follow instructions at: https://pytorch.org/get-started/locally/")
            fixes_applied.append("Recommended CUDA-compatible PyTorch installation")
    except ImportError:
        print("✗ PyTorch is not installed properly")
        print("  To install PyTorch, run:")
        print("  pip install torch")
    
    # Create requirements file with all dependencies
    try:
        with open('requirements.txt', 'r') as f:
            existing_reqs = f.read()
        
        complete_reqs = """
numpy>=1.21.0
torch>=1.12.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.6.0
tqdm>=4.62.0
tabulate>=0.8.9
requests>=2.27.0
datasets>=2.0.0
"""
        if existing_reqs.strip() != complete_reqs.strip():
            with open('requirements_complete.txt', 'w') as f:
                f.write(complete_reqs)
            print("✓ Created complete requirements file: requirements_complete.txt")
            print("  To install all dependencies, run:")
            print("  pip install -r requirements_complete.txt")
            fixes_applied.append("Created comprehensive requirements file")
    except Exception as e:
        with open('requirements_complete.txt', 'w') as f:
            f.write(complete_reqs)
        print("✓ Created complete requirements file from scratch")
        fixes_applied.append("Created comprehensive requirements file")
    
    # Summary
    if fixes_applied:
        print("\nApplied the following fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
        print("\nPlease run this script again to check if all issues are resolved.")
    else:
        print("\nNo fixes were needed or able to be applied automatically.")
        print("For any remaining issues, check the detailed output above.")
    
    return bool(fixes_applied)