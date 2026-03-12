import sys
import traceback

try:
    import pickle
    import numpy as np
    from pathlib import Path
    import tensorflow as tf
    from tensorflow import keras
    
    print("\n" + "="*70)
    print("🎥 TRAINING VISION MODEL - DEBUG MODE")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "datasets" / "combined"
    
    print(f"\n📂 Looking for data in: {data_path}")
    
    # Check if files exist
    train_file = data_path / "train" / "urfall_video_data.pkl"
    val_file = data_path / "val" / "urfall_video_data.pkl"
    test_file = data_path / "test" / "urfall_video_data.pkl"
    
    print(f"\n🔍 Checking files:")
    print(f"   Train: {train_file.exists()} - {train_file}")
    print(f"   Val:   {val_file.exists()} - {val_file}")
    print(f"   Test:  {test_file.exists()} - {test_file}")
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        print("\n❌ Some files are missing!")
        print("   Please run: python scripts/preprocess_urfall.py")
        sys.exit(1)
    
    # Load data
    print("\n📂 Loading data...")
    
    print("   Loading training data...")
    with open(train_file, 'rb') as f:
        train_data_dict = pickle.load(f)
    print(f"   ✓ Train data loaded")
    
    print("   Loading validation data...")
    with open(val_file, 'rb') as f:
        val_data_dict = pickle.load(f)
    print(f"   ✓ Val data loaded")
    
    print("   Loading test data...")
    with open(test_file, 'rb') as f:
        test_data_dict = pickle.load(f)
    print(f"   ✓ Test data loaded")
    
    # Extract arrays
    print("\n📊 Extracting data arrays...")
    X_train = train_data_dict['data']
    y_train = train_data_dict['labels']
    X_val = val_data_dict['data']
    y_val = val_data_dict['labels']
    X_test = test_data_dict['data']
    y_test = test_data_dict['labels']
    
    print(f"\n✓ Data loaded successfully!")
    print(f"   Train: {len(X_train)} sequences, shape: {X_train.shape}")
    print(f"   Val:   {len(X_val)} sequences, shape: {X_val.shape}")
    print(f"   Test:  {len(X_test)} sequences, shape: {X_test.shape}")
    
    print("\n✅ All checks passed! Data is ready for training.")
    print("\n💡 If you see this message, the issue is in the model building/training part.")
    print("   I'll create a simpler training script for you.")
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR OCCURRED!")
    print("="*70)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    print("-"*70)
    traceback.print_exc()
    print("-"*70)
    print("\n💡 Please copy and paste the error above so I can help fix it!")