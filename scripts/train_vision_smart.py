import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime
pip install numpy pandas scikit-learn tensorflow matplotlib opencv-python tqdm

def find_urfall_data():
    """Find UR Fall data files with any naming"""
    project_root = Path(__file__).parent.parent
    combined_path = project_root / "datasets" / "combined"
    
    # Possible file names
    possible_names = ['urfall_video_data.pkl', 'urfall_data.pkl']
    
    files = {}
    for split in ['train', 'val', 'test']:
        split_path = combined_path / split
        found = False
        
        for name in possible_names:
            file_path = split_path / name
            if file_path.exists():
                files[split] = file_path
                found = True
                break
        
        if not found:
            # Try to find ANY urfall file
            urfall_files = list(split_path.glob('urfall*.pkl'))
            if urfall_files:
                files[split] = urfall_files[0]
    
    return files

def build_vision_model(input_shape):
    """Build CNN-BiLSTM model for temporal vision analysis"""
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # TimeDistributed CNN layers
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Dropout(0.25)),
        
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Dropout(0.25)),
        
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Dropout(0.25)),
        
        layers.TimeDistributed(layers.Flatten()),
        
        # Bidirectional LSTM
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.4),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_vision_model():
    """Train vision model with smart file detection"""
    
    print("\n" + "="*70)
    print("🎥 TRAINING VISION MODEL (CNN-BiLSTM) - SMART VERSION")
    print("="*70)
    
    # Find data files
    print("\n🔍 Looking for UR Fall data files...")
    files = find_urfall_data()
    
    if len(files) != 3:
        print(f"\n❌ Could not find all data files!")
        print(f"   Found: {list(files.keys())}")
        print(f"   Need: train, val, test")
        return
    
    print("\n✓ Found data files:")
    for split, filepath in files.items():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   {split:5s}: {filepath.name} ({size_mb:.2f} MB)")
    
    # Paths
    project_root = Path(__file__).parent.parent
    results_path = project_root / "results"
    models_path = results_path / "models"
    
    models_path.mkdir(parents=True, exist_ok=True)
    (results_path / "figures").mkdir(parents=True, exist_ok=True)
    (results_path / "metrics").mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    
    with open(files['train'], 'rb') as f:
        train_dict = pickle.load(f)
    with open(files['val'], 'rb') as f:
        val_dict = pickle.load(f)
    with open(files['test'], 'rb') as f:
        test_dict = pickle.load(f)
    
    X_train = train_dict['data']
    y_train = train_dict['labels']
    X_val = val_dict['data']
    y_val = val_dict['labels']
    X_test = test_dict['data']
    y_test = test_dict['labels']
    
    print(f"✓ Data loaded successfully")
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Val:   {len(X_val)} sequences")
    print(f"   Test:  {len(X_test)} sequences")
    print(f"\n📐 Data shape: {X_train.shape}")
    
    # Build model
    print("\n🏗️ Building CNN-BiLSTM model...")
    input_shape = X_train.shape[1:]
    model = build_vision_model(input_shape)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print("✓ Model built")
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.000001,
        verbose=1
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n✅ Training complete!")
    
    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATING ON TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    print(f"\n📈 Test Results:")
    print(f"   Accuracy:  {test_acc*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall:    {test_recall*100:.2f}%")
    print(f"   F1-Score:  {test_f1*100:.2f}%")
    print(f"   Loss:      {test_loss:.4f}")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print("-"*70)
    report = classification_report(y_test, y_pred, target_names=['ADL', 'Fall'], digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                ADL    Fall")
    print(f"Actual  ADL    {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"        Fall   {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    # Save
    print("\n💾 Saving model and results...")
    model_file = models_path / f"vision_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(model_file)
    print(f"✓ Model saved: {model_file.name}")
    
    metrics = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1_score': float(test_f1),
        'test_loss': float(test_loss),
        'confusion_matrix': cm.tolist(),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_file = results_path / "metrics" / f"vision_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {metrics_file.name}")
    
    # Plots
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = results_path / "figures" / f"vision_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Vision Model')
    plt.colorbar()
    plt.xticks([0, 1], ['ADL', 'Fall'])
    plt.yticks([0, 1], ['ADL', 'Fall'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot_file = results_path / "figures" / f"vision_cm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved")
    
    print("\n" + "="*70)
    print("✅ VISION MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📋 Summary:")
    print(f"   • Model accuracy: {test_acc*100:.2f}%")
    print(f"   • F1-Score: {test_f1*100:.2f}%")
    print(f"   • Model file: {model_file.name}")
    print("\n")
    
    return model, metrics

if __name__ == "__main__":
    train_vision_model()