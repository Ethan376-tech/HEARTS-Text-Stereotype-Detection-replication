import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HEARTS PROJECT - INSTALLATION VERIFICATION")
print("="*70)
print(f"\nPython version: {sys.version.split()[0]}")
print(f"Python path: {sys.executable}\n")

# Test all packages
test_results = []

def test_import(package_name, display_name, check_version=True):
    try:
        if package_name == 'sklearn':
            import sklearn
            version = sklearn.__version__
        elif package_name == 'spacy_model':
            import spacy
            nlp = spacy.load("en_core_web_lg")
            version = f"Model loaded ({nlp.meta['version']})"
        else:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'installed')
        
        status = "✓"
        test_results.append(True)
        print(f"{status} {display_name:25s} {version}")
        return True
    except Exception as e:
        print(f"✗ {display_name:25s} ERROR: {str(e)[:40]}")
        test_results.append(False)
        return False

# Core packages
print("\n" + "-"*70)
print("CORE PACKAGES")
print("-"*70)
test_import('numpy', 'NumPy')
test_import('scipy', 'SciPy')
test_import('torch', 'PyTorch')
test_import('transformers', 'Transformers')
test_import('datasets', 'Datasets')

# Machine Learning
print("\n" + "-"*70)
print("MACHINE LEARNING")
print("-"*70)
test_import('sklearn', 'Scikit-learn')
test_import('pandas', 'Pandas')

# Visualization
print("\n" + "-"*70)
print("VISUALIZATION")
print("-"*70)
test_import('matplotlib', 'Matplotlib')
test_import('seaborn', 'Seaborn')

# Explainability
print("\n" + "-"*70)
print("EXPLAINABILITY")
print("-"*70)
test_import('shap', 'SHAP')
test_import('lime', 'LIME')

# NLP
print("\n" + "-"*70)
print("NLP TOOLS")
print("-"*70)
test_import('spacy', 'SpaCy')
test_import('spacy_model', 'SpaCy Model (en_core_web_lg)')

# Utilities
print("\n" + "-"*70)
print("UTILITIES")
print("-"*70)
test_import('tqdm', 'TQDM')
test_import('codecarbon', 'CodeCarbon')
test_import('huggingface_hub', 'Hugging Face Hub')
test_import('accelerate', 'Accelerate')
test_import('sentencepiece', 'SentencePiece')

# Hardware check
print("\n" + "-"*70)
print("HARDWARE")
print("-"*70)
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"🖥️  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   Using CPU (this is fine for this project)")
except Exception as e:
    print(f"   Error checking CUDA: {e}")

# Summary
print("\n" + "="*70)
success_rate = sum(test_results) / len(test_results) * 100
if all(test_results):
    print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
    print("="*70)
    print("\n📋 NEXT STEPS:")
    print("   1. python create_folders.py")
    print("   2. python download_dataset.py")
    print("   3. python download_model.py")
    print("   4. python test_baseline_model.py")
else:
    print(f"⚠️  INSTALLATION {success_rate:.0f}% COMPLETE")
    print("="*70)
    print("\n❌ Some packages failed. Check errors above.")

print("\n" + "="*70)