import torch
from PIL import Image
from torchvision import transforms
from models.image.classification.MedViT.MedViT import MedViT_small
import medmnist
from medmnist import INFO
import numpy as np

# Human-readable label maps for MedMNIST datasets
MEDMNIST_LABELS = {
    "bloodmnist": {
        "0": "neutrophil",
        "1": "eosinophil",
        "2": "basophil",
        "3": "lymphocyte",
        "4": "monocyte"
    },
    "breastmnist": {
        "0": "normal",
        "1": "malignant"
    },
    "chestmnist": {
        "0": "No Finding",
        "1": "Enlarged Cardiomediastinum",
        "2": "Cardiomegaly",
        "3": "Lung Opacity",
        "4": "Lung Lesion",
        "5": "Edema",
        "6": "Consolidation",
        "7": "Pneumonia",
        "8": "Atelectasis",
        "9": "Pneumothorax",
        "10": "Pleural Effusion",
        "11": "Pleural Other",
        "12": "Fracture",
        "13": "Support Devices"
    },
    "dermamnist": {
        "0": "Actinic keratoses",
        "1": "Basal cell carcinoma",
        "2": "Benign keratosis-like lesions",
        "3": "Dermatofibroma",
        "4": "Melanocytic nevi",
        "5": "Melanoma",
        "6": "Vascular lesions"
    },
    "octmnist": {
        "0": "CNV",
        "1": "DME",
        "2": "Drusen",
        "3": "Normal"
    },
    "organamnist": {
        "0": "spleen",
        "1": "lung",
        "2": "kidney",
        "3": "liver",
        "4": "brain",
        "5": "prostate",
        "6": "bowel",
        "7": "muscle",
        "8": "eye"
    },
    "organcmnist": {
        "0": "colon",
        "1": "esophagus",
        "2": "kidney",
        "3": "lung",
        "4": "prostate",
        "5": "stomach",
        "6": "uterus"
    },
    "organsmnist": {
        "0": "spleen",
        "1": "lung",
        "2": "kidney",
        "3": "liver"
    },
    "pathmnist": {
        "0": "adipose",
        "1": "background",
        "2": "debris",
        "3": "lymphocytes",
        "4": "mucus",
        "5": "smooth muscle",
        "6": "normal colon mucosa",
        "7": "cancer-associated stroma",
        "8": "colorectal adenocarcinoma epithelium"
    },
    "pneumoniamnist": {
        "0": "normal",
        "1": "pneumonia"
    }
}

def to_rgb(image):
    return image.convert('RGB')

class MedViTClassifier:
    _MODEL_CACHE = {}

    @staticmethod
    def preload_all_models(weights_dir="models/image/classification/Models"):
        import os
        ALLOWED_DATASETS = [
            'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
            'octmnist', 'organamnist', 'organcmnist',
            'organsmnist', 'pathmnist', 'pneumoniamnist',
        ]
        for dataset in ALLOWED_DATASETS:
            if dataset in MedViTClassifier._MODEL_CACHE:
                continue
            weights_path = os.path.join(weights_dir, f"{dataset}_weights.pkl")
            if os.path.exists(weights_path):
                MedViTClassifier._MODEL_CACHE[dataset] = MedViTClassifier(data_flag=dataset, weights_path=weights_path)
            else:
                print(f"[Startup] Weights not found for {dataset}, skipping preload.")


    @staticmethod
    def run_all_models(image_path_or_pil, weights_dir="models/image/classification/Models"):
        """
        Run the input image through all available trained MedViT models (for each ALLOWED_DATASET).
        Returns a dict: {dataset_name: prediction_dict}
        """
        import os
        results = {}
        ALLOWED_DATASETS = [
            'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
            'octmnist', 'organamnist', 'organcmnist',
            'organsmnist', 'pathmnist', 'pneumoniamnist', 
        ]
        for dataset in ALLOWED_DATASETS:
            if dataset not in MedViTClassifier._MODEL_CACHE:
                continue  # skip missing models
            try:
                model = MedViTClassifier._MODEL_CACHE[dataset]
                result = model.predict(image_path_or_pil)
                pred_label = result.get("predicted_label")
                probabilities = result.get("probabilities")
                if pred_label is not None and probabilities is not None:
                    # Only return the probability for the predicted label
                    pred_class = probabilities.index(max(probabilities)) if hasattr(probabilities, 'index') else int(np.argmax(probabilities))
                    single_prob = probabilities[pred_class]
                    results[dataset] = {
                        "predicted_label": pred_label,
                        "probability": single_prob
                    }
            except Exception:
                continue  # filter out errors
        return results



    def __init__(self, data_flag='retinamnist', weights_path=None):
        self.data_flag = data_flag  # Store dataset name for correct label mapping
        # Only allow datasets with textual (non-numeric) labels (hardcoded)
        ALLOWED_DATASETS = [
            'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
            'octmnist', 'organamnist', 'organcmnist',
            'organsmnist', 'pathmnist', 'pneumoniamnist', 
        ]
        if data_flag not in ALLOWED_DATASETS:
            raise ValueError(f"Dataset '{data_flag}' is not allowed: only datasets with textual (non-numeric) labels are supported. Allowed: {ALLOWED_DATASETS}")
        # Set up dataset info
        info = INFO[data_flag]
        self.n_classes = len(info['label'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MedViTClassifier] Using device: {self.device}")
        if self.device.type == 'cpu':
            print("[MedViTClassifier][WARNING] CUDA GPU not detected! Training will be slow. If you have a GPU, check your PyTorch installation and drivers.")

        # Always use MedViT_small
        self.model = MedViT_small(num_classes=self.n_classes)
        self.model.to(self.device)
        self.model.eval()

        # Define the transform for inference and training
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # Load or train weights
        import os
        if weights_path is None:
            weights_path = f'models/image/classification/Models/{data_flag}_weights.pkl'  # Save as .pkl
        if not os.path.exists(weights_path):
            print(f"Weights file {weights_path} not found. Training model from scratch...")
            self._train_and_save_model(weights_path, data_flag)
        else:
            print(f"Loading model weights from {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def _train_and_save_model(self, weights_path, data_flag):
        # Follow Instructions.ipynb: use medmnist loader, train for a few epochs, save weights
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import medmnist
        from medmnist import Evaluator, INFO
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])
        # Use the same transform as for inference
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        train_dataset = DataClass(split='train', download=True, transform=transform)
        test_dataset = DataClass(split='test', download=True, transform=transform)
        # Limit training set to max 5000 samples for speed
        from torch.utils.data import Subset
        if len(train_dataset) > 500:
            train_dataset = Subset(train_dataset, range(500))
        train_loader = DataLoader(train_dataset, num_workers=10,pin_memory=True, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, num_workers=10,pin_memory=True, batch_size=16, shuffle=False)
        # Choose loss and label type based on task
        if info['task'] == 'multi-label, binary-class':
            criterion = torch.nn.BCEWithLogitsLoss()
            is_multilabel = True
        else:
            criterion = torch.nn.CrossEntropyLoss()
            is_multilabel = False
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        device = self.device
        self.model.train()
        try:
            for epoch in range(3):  # Limit to 3 epochs
                running_loss = 0.0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.to(self.device, non_blocking=True)
                    if is_multilabel:
                        labels = labels.to(self.device, non_blocking=True).float()
                    else:
                        labels = labels.squeeze().long().to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    if is_multilabel:
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if batch_idx % 20 == 0:
                        print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}")
                print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory! Try reducing batch size further or use CPU for training.")
            raise
        print(f"Saving trained model weights to {weights_path}")
        import os
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), weights_path)
        self.model.eval()


    def predict(self, image_path_or_pil):
        # Accepts file path or PIL image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
        # Get label names from INFO using self.data_flag
        ds_name = self.data_flag
        # Use hardcoded human-readable labels if available
        if ds_name in MEDMNIST_LABELS:
            label_dict = MEDMNIST_LABELS[ds_name]
            label_map = [label_dict[str(i)] for i in range(self.n_classes)]
        else:
            info = INFO[ds_name]
            label_dict = info['label'] if 'label' in info else {str(i): str(i) for i in range(self.n_classes)}
            label_map = [label_dict[str(i)] for i in range(self.n_classes)]
        return {
            "predicted_label": label_map[pred_class],
            "probabilities": probs.tolist()
        }