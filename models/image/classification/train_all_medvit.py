"""
Train MedViTClassifier on all allowed MedMNIST datasets with textual labels.
Run: python train_all_medvit.py
"""
from medvit_classifier import MedViTClassifier

ALLOWED_DATASETS = [
    'dermamnist', 'pathmnist'  
]

if __name__ == "__main__":
    trained = []
    already_trained = []
    failed = []
    import os
    weights_dir = "models/image/classification/Models"
    os.makedirs(weights_dir, exist_ok=True)
    for flag in ALLOWED_DATASETS:
        print(f"\n===== Training MedViTClassifier for dataset: {flag} =====")
        weights_path = os.path.join(weights_dir, f"{flag}_weights.pkl")
        if os.path.exists(weights_path):
            print(f"Weights already exist for {flag}, skipping training.")
            already_trained.append(flag)
            continue
        try:
            # Create an instance directly instead of using get_instance
            classifier = MedViTClassifier(data_flag=flag, weights_path=weights_path)
            # Train the model
            classifier.train()
            # Save the model
            classifier.save_model()
            trained.append(flag)
            print(f"Successfully trained {flag}.")
        except Exception as e:
            print(f"Failed to train {flag}: {e}")
            failed.append(flag)
    print("\n=== Training Summary ===")
    print(f"Trained: {trained}")
    print(f"Already trained (skipped): {already_trained}")
    print(f"Failed: {failed}")
