from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

print("="*70)
print("DOWNLOADING ALBERT-V2 BIAS CLASSIFIER")
print("="*70)

model_name = "holistic-ai/bias_classifier_albertv2"

try:
    print(f"\n📥 Downloading model: {model_name}")
    print("This may take 5-10 minutes...\n")
    
    # Download tokenizer
    print("   Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Download model
    print("   Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Save locally
    save_path = "./models/bias_classifier_albertv2"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n💾 Saving to '{save_path}'...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Model info
    print(f"\n📊 Model Information:")
    print(f"   Model type:          {model.config.model_type}")
    print(f"   Parameters:          {model.num_parameters():,}")
    print(f"   Hidden size:         {model.config.hidden_size}")
    print(f"   Number of layers:    {model.config.num_hidden_layers}")
    print(f"   Attention heads:     {model.config.num_attention_heads}")
    print(f"   Number of labels:    {model.config.num_labels}")
    print(f"   Max sequence length: {tokenizer.model_max_length}")
    
    # Verify save
    size_mb = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(save_path)
        for filename in filenames
    ) / (1024 * 1024)
    
    print(f"\n   Model size: {size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ MODEL DOWNLOAD COMPLETE!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Check internet connection")
    print("   2. Try: pip install --upgrade transformers")
    print("   3. Ensure ~100MB free disk space")