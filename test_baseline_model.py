from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HEARTS BASELINE MODEL EVALUATION")
print("="*70)

# Load model and tokenizer
print("\n📥 Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained("./models/bias_classifier_albertv2")
tokenizer = AutoTokenizer.from_pretrained("./models/bias_classifier_albertv2")

# Load dataset
print("📥 Loading test dataset...")
dataset = load_from_disk("./data/EMGSD")
test_data = dataset['test']
print(f"   Test set size: {len(test_data):,} samples")

# Get label mapping from model config
label2id = model.config.label2id
print(f"   Label mapping: {label2id}")

# Check a few sample labels to understand the format
print("\n   Checking sample labels from dataset...")
for i in range(min(5, len(test_data))):
    sample_label = test_data[i]['label']
    print(f"   Sample {i}: label = {repr(sample_label)} (type: {type(sample_label).__name__})")

# Device setup with detailed diagnostics
print("\n🖥️  Checking GPU availability...")
cuda_available = torch.cuda.is_available()

# Check if PyTorch was built with CUDA support
try:
    cuda_built = torch.version.cuda is not None
except:
    cuda_built = False

print(f"   PyTorch built with CUDA: {cuda_built}")
print(f"   CUDA available: {cuda_available}")

if cuda_available:
    print(f"   GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   CUDA version: {torch.version.cuda}")
    device = torch.device("cuda:0")
    print(f"   ✅ Using GPU: {device}")
elif cuda_built:
    print("   ⚠️  PyTorch has CUDA support but GPU not detected")
    print("      - Check if NVIDIA drivers are installed")
    print("      - Run: nvidia-smi to verify GPU is visible")
    device = torch.device("cpu")
    print(f"   Using CPU: {device}")
else:
    print("   ⚠️  PyTorch installed without CUDA support (CPU-only version)")
    print("   💡 To use GPU, install PyTorch with CUDA support:")
    print("      For CUDA 11.8: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("      For CUDA 12.1: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("      Or check: https://pytorch.org/get-started/locally/")
    device = torch.device("cpu")
    print(f"   Using CPU: {device}")

model.to(device)
model.eval()

# Quick test on 100 samples
print("\n" + "="*70)
print("QUICK TEST: First 100 Samples")
print("="*70)

predictions = []
true_labels = []

print("\nRunning predictions...")
for i in tqdm(range(min(100, len(test_data)))):
    text = test_data[i]['text']
    label = test_data[i]['label']
    
    # Convert label to integer
    # Handle different label formats from the dataset
    if isinstance(label, (list, tuple)):
        # If label is a list, take the first element
        label = label[0] if len(label) > 0 else 'neutral'
    
    if isinstance(label, str):
        label_lower = label.lower().strip()
        # Check if label contains "stereotype" (case-insensitive)
        if 'stereotype' in label_lower:
            label = 1  # Stereotype
        elif 'non-stereotype' in label_lower or 'nonstereotype' in label_lower or label_lower == 'neutral':
            label = 0  # Non-Stereotype
        elif label.isdigit():
            label = int(label)
        else:
            # Try to map using label2id, default to 0 if not found
            label = label2id.get(label, 0)
    else:
        label = int(label)  # Already a number
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    predictions.append(pred)
    true_labels.append(label)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1_macro = f1_score(true_labels, predictions, average='macro')
f1_binary = f1_score(true_labels, predictions, average='binary')

print("\n📊 RESULTS (100 samples):")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Macro F1 Score:  {f1_macro:.4f} ({f1_macro*100:.2f}%)")
print(f"   Binary F1 Score: {f1_binary:.4f} ({f1_binary*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(true_labels, predictions, 
                          target_names=['Non-Stereotype', 'Stereotype'],
                          digits=4))

print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(true_labels, predictions)
print(f"                  Predicted")
print(f"                  0      1")
print(f"   Actual 0    {cm[0][0]:4d}   {cm[0][1]:4d}")
print(f"          1    {cm[1][0]:4d}   {cm[1][1]:4d}")

# Compare with paper
paper_f1 = 0.815
difference = abs(f1_macro - paper_f1)
percentage_diff = (difference / paper_f1) * 100

print("\n" + "="*70)
print("COMPARISON WITH PAPER")
print("="*70)
print(f"   Paper F1 Score (EMGSD):          {paper_f1:.4f} (81.5%)")
print(f"   Your F1 Score (100 samples):     {f1_macro:.4f} ({f1_macro*100:.2f}%)")
print(f"   Difference:                      {difference:.4f} ({percentage_diff:.2f}%)")

if percentage_diff <= 5:
    print("\n   ✅ REPLICATION SUCCESSFUL! (within ±5%)")
else:
    print(f"\n   ⚠️  Results differ by {percentage_diff:.2f}%")
    print("   Note: Small sample - full test recommended")

# Run full test automatically
print("\n" + "="*70)
print("\n🔍 Running full dataset test")
print(f"   Will test all {len(test_data):,} samples")
print("   Estimated time: 10-30 minutes (CPU) or 3-5 minutes (GPU)")

# Automatically run full test
if True:
    print("\n" + "="*70)
    print("FULL TEST SET EVALUATION")
    print("="*70)
    
    predictions_full = []
    true_labels_full = []
    
    print(f"\nTesting {len(test_data):,} samples...")
    for i in tqdm(range(len(test_data))):
        text = test_data[i]['text']
        label = test_data[i]['label']
        
        # Convert label to integer
        # Handle different label formats from the dataset
        if isinstance(label, (list, tuple)):
            # If label is a list, take the first element
            label = label[0] if len(label) > 0 else 'neutral'
        
        if isinstance(label, str):
            label_lower = label.lower().strip()
            # Check if label contains "stereotype" (case-insensitive)
            if 'stereotype' in label_lower:
                label = 1  # Stereotype
            elif 'non-stereotype' in label_lower or 'nonstereotype' in label_lower or label_lower == 'neutral':
                label = 0  # Non-Stereotype
            elif label.isdigit():
                label = int(label)
            else:
                # Try to map using label2id, default to 0 if not found
                label = label2id.get(label, 0)
        else:
            label = int(label)  # Already a number
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        predictions_full.append(pred)
        true_labels_full.append(label)
    
    # Calculate full metrics
    accuracy_full = accuracy_score(true_labels_full, predictions_full)
    f1_macro_full = f1_score(true_labels_full, predictions_full, average='macro')
    f1_binary_full = f1_score(true_labels_full, predictions_full, average='binary')
    
    print("\n📊 FINAL RESULTS (Full Test Set):")
    print(f"   Accuracy:        {accuracy_full:.4f} ({accuracy_full*100:.2f}%)")
    print(f"   Macro F1 Score:  {f1_macro_full:.4f} ({f1_macro_full*100:.2f}%)")
    print(f"   Binary F1 Score: {f1_binary_full:.4f} ({f1_binary_full*100:.2f}%)")
    
    print("\n📋 Classification Report:")
    print(classification_report(true_labels_full, predictions_full,
                              target_names=['Non-Stereotype', 'Stereotype'],
                              digits=4))
    
    # Final comparison
    difference_full = abs(f1_macro_full - paper_f1)
    percentage_diff_full = (difference_full / paper_f1) * 100
    
    print("\n" + "="*70)
    print("FINAL COMPARISON WITH PAPER")
    print("="*70)
    print(f"   Paper F1 Score:              {paper_f1:.4f} (81.5%)")
    print(f"   Your F1 Score:               {f1_macro_full:.4f} ({f1_macro_full*100:.2f}%)")
    print(f"   Absolute Difference:         {difference_full:.4f}")
    print(f"   Percentage Difference:       {percentage_diff_full:.2f}%")
    
    if percentage_diff_full <= 5:
        print("\n   🎉 ✅ BASELINE REPLICATION SUCCESSFUL!")
        print("   Your results are within ±5% of the paper.")
    else:
        print(f"\n   ⚠️  Results differ by {percentage_diff_full:.2f}%")
    
    # Save results
    result_file = "./results/baseline/baseline_results.txt"
    with open(result_file, "w") as f:
        f.write("HEARTS BASELINE REPLICATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Paper F1 Score (EMGSD):     {paper_f1:.4f} (81.5%)\n")
        f.write(f"Your F1 Score:              {f1_macro_full:.4f} ({f1_macro_full*100:.2f}%)\n")
        f.write(f"Absolute Difference:        {difference_full:.4f}\n")
        f.write(f"Percentage Difference:      {percentage_diff_full:.2f}%\n")
        f.write(f"Accuracy:                   {accuracy_full:.4f}\n")
        f.write(f"Binary F1:                  {f1_binary_full:.4f}\n\n")
        f.write("Status: " + ("SUCCESSFUL ✓" if percentage_diff_full <= 5 else "REVIEW NEEDED") + "\n")
    
    print(f"\n💾 Results saved to: {result_file}")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)