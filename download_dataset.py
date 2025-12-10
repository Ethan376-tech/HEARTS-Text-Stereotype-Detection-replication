from datasets import load_dataset
import os

print("="*70)
print("DOWNLOADING EMGSD DATASET")
print("="*70)

try:
    print("\n📥 Connecting to Hugging Face...")
    dataset = load_dataset("holistic-ai/EMGSD")
    
    print("\n✅ Dataset loaded successfully!")
    print(f"\n📊 Dataset Structure:")
    print(f"   Train samples: {len(dataset['train']):,}")
    print(f"   Test samples:  {len(dataset['test']):,}")
    print(f"   Total:         {len(dataset['train']) + len(dataset['test']):,}")
    
    # Show sample
    print(f"\n📋 Sample entry from training set:")
    sample = dataset['train'][0]
    print(f"   Text: {sample['text'][:80]}...")
    print(f"   Label: {sample['label']}")
    print(f"   Stereotype Type: {sample['stereotype_type']}")
    
    # Show features
    print(f"\n📝 Dataset Features:")
    for feature, dtype in dataset['train'].features.items():
        print(f"   - {feature:20s} {dtype}")
    
    # Save locally
    save_path = "./data/EMGSD"
    print(f"\n💾 Saving dataset to '{save_path}'...")
    dataset.save_to_disk(save_path)
    
    # Verify save
    size_mb = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(save_path)
        for filename in filenames
    ) / (1024 * 1024)
    
    print(f"   Dataset size: {size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ DATASET DOWNLOAD COMPLETE!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Check internet connection")
    print("   2. Try: pip install --upgrade huggingface-hub")
    print("   3. Ensure ~150MB free disk space")