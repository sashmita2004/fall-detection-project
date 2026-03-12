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

def pad_sequences(sequences, max_length=None):
    """Pad sequences to same length"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = np.zeros((len(sequences), max_length, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length, :] = seq[:length]
    
    return padded

def build_sensor_model(input_shape):
    """Build sensor fusion model with data augmentation"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # CNN layers for feature extraction
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # LSTM layers for temporal patterns
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(32),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_sensor_model():
    """Train sensor fusion model with real SisFall data"""
    
    print("\n" + "="*70)
    print("🤖 TRAINING SENSOR FUSION MODEL WITH REAL DATA")
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
    print("\n📂 Loading data...")
    
    with open(data_path / "train" / "sisfall_data.pkl", 'rb') as f:
        train_data_dict = pickle.load(f)
    
    with open(data_path / "val" / "sisfall_data.pkl", 'rb') as f:
        val_data_dict = pickle.load(f)
    
    with open(data_path / "test" / "sisfall_data.pkl", 'rb') as f:
        test_data_dict = pickle.load(f)
    
    print("✓ Data loaded successfully")
    
    # Extract data
    X_train = train_data_dict['data']
    y_train = train_data_dict['labels']
    X_val = val_data_dict['data']
    y_val = val_data_dict['labels']
    X_test = test_data_dict['data']
    y_test = test_data_dict['labels']
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Pad sequences to same length
    print("\n🔧 Padding sequences to uniform length...")
    max_length = 200  # Use fixed length for consistency
    
    X_train_padded = pad_sequences(X_train, max_length)
    X_val_padded = pad_sequences(X_val, max_length)
    X_test_padded = pad_sequences(X_test, max_length)
    
    print(f"✓ Padded shape: {X_train_padded.shape}")
    
    # Build model
    print("\n🏗️ Building model...")
    input_shape = (max_length, 6)  # 6 features: AccX, AccY, AccZ, GyroX, GyroY, GyroZ
    model = build_sensor_model(input_shape)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print("✓ Model built")
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Train model
    print("\n🚀 Starting training...")
    print("="*70)
    
    history = model.fit(
        X_train_padded, y_train,
        validation_data=(X_val_padded, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n✅ Training complete!")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("📊 EVALUATING ON TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        X_test_padded, y_test, verbose=0
    )
    
    print(f"\n📈 Test Results:")
    print(f"   Accuracy: {test_acc*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall: {test_recall*100:.2f}%")
    print(f"   F1-Score: {2*(test_precision*test_recall)/(test_precision+test_recall)*100:.2f}%")
    print(f"   Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred_proba = model.predict(X_test_padded, verbose=0)
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
    model_file = models_path / f"sensor_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(model_file)
    print(f"✓ Model saved to: {model_file}")
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1_score': float(2*(test_precision*test_recall)/(test_precision+test_recall)),
        'test_loss': float(test_loss),
        'confusion_matrix': cm.tolist(),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_file = results_path / "metrics" / f"sensor_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Plot training history
    print("\n📊 Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = results_path / "figures" / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {plot_file}")
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Sensor Model')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['ADL', 'Fall'])
    plt.yticks(tick_marks, ['ADL', 'Fall'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot_file = results_path / "figures" / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix plot saved to: {cm_plot_file}")
    
    print("\n" + "="*70)
    print("✅ SENSOR MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\n📋 Summary:")
    print(f"   • Model accuracy: {test_acc*100:.2f}%")
    print(f"   • F1-Score: {2*(test_precision*test_recall)/(test_precision+test_recall)*100:.2f}%")
    print(f"   • Model saved: {model_file.name}")
    print(f"   • Metrics saved: {metrics_file.name}")
    print(f"   • Plots saved in: results/figures/")
    print("\n💡 Next: You can now generate your project report!")
    print("   This model achieved real results on real data!")
    print("\n")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = train_sensor_model()