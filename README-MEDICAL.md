from nexus_medical_integration import apply_medical_patches

# Apply the patches for medical data
apply_medical_patches()

# Load custom medical data
X, y, feature_names = load_my_medical_dataset()

# Create data loaders
train_loader, test_loader = create_data_loaders(X, y)

# Initialize knowledge graph
kg = create_medical_knowledge_graph(feature_names)

# Run the experiment
result = run_scaled_nexus_experiment(
    custom_train_loader=train_loader,
    custom_test_loader=test_loader,
    custom_kg=kg,
    feature_names=feature_names,
    class_names=class_names
)