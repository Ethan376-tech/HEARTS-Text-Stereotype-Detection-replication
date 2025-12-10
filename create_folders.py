import os

folders = [
    './data',
    './models',
    './results',
    './notebooks',
    './results/figures',
    './results/baseline',
    './results/adapted'
]

print("="*70)
print("Creating project folders...")
print("="*70)

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ Created: {folder}")

print("\n✅ All folders created successfully!")
print("="*70)