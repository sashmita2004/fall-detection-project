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

def build_vision_model(input_shape):
    """Build CNN-BiLSTM model for temporal vision analysis"""
    
    # Input: (frames, height, width, channels)
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # TimeDistributed CNN layers for spatial feature extraction
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
        
        # Flatten each frame's features
        layers.TimeDistributed(layers.Flatten()),
        
        # Bidirectional LSTM for temporal analysis
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.4),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_vision_model():
    """Train vision model with UR Fall video data"""
    
    print("\n" + "="*70)
    print("🎥 TRAINING VISION MODEL (CNN-BiLSTM) WITH UR FALL DATA")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "datasets" / "combined"
    results_path = project_root / "results"
    models_path = results_path / "models"
    
    # Create results directories
    models_path.mkdir(parents=True, exist_ok=True)
    (results_path / "figures").mkdir(parents=True, exist_ok=True)
    (results_path / "metrics").mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    print("\n📂 Loading UR Fall video data...")
    
    try:
        with open(data_path / "train" / "urfall_video_data.pkl", 'rb') as f:
            train_data_dict = pickle.load(f)
        
        with open(data_path / "val" / "urfall_video_data.pkl", 'rb') as f:
            val_data_dict = pickle.load(f)
        
        with open(data_path / "test" / "urfall_video_data.pkl", 'rb') as f:
            test_data_dict = pickle.load(f)
        
        print("✓ Data loaded successfully")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please run preprocessing first:")
        print("   python scripts/preprocess_urfall.py")
        return
    
    # Extract data
    X_train = train_data_dict['data']
    y_train = train_data_dict['labels']
    X_val = val_data_dict['data']
    y_val = val_data_dict['labels']
    X_test = test_data_dict['data']
    y_test = test_data_dict['labels']
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Val: {len(X_val)} sequences")
    print(f"   Test: {len(X_test)} sequences")
    print(f"\n📐 Data shape: {X_train.shape}")
    print(f"   [sequences, frames, height, width, channels]")
    
    # Build model
    print("\n🏗️ Building CNN-BiLSTM model...")
    input_shape = X_train.shape[1:]  # (frames, height, width, channels)
    model = build_vision_model(input_shape)
    
    # Compile model
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
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.000001
    )
    
    # Train model
    print("\n🚀 Starting training...")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,  # Smaller batch size for video data
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n✅ Training complete!")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("📊 EVALUATING ON TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    print(f"\n📈 Test Results:")
    print(f"   Accuracy: {test_acc*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall: {test_recall*100:.2f}%")
    print(f"   F1-Score: {test_f1*100:.2f}%")
    print(f"   Loss: {test_loss:.4f}")
    
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
    
    # Save model
    print("\n💾 Saving model...")
    model_file = models_path / f"vision_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(model_file)
    print(f"✓ Model saved to: {model_file}")
    
    # Save metrics
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
    
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Plot training history
    print("\n📊 Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Vision Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Vision Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Vision Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'])