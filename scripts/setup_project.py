import sys
from pathlib import Path

# Add the parent folder to Python's search path
sys.path.append(str(Path(__file__).parent.parent))

# Import our config file
from config.dataset_config import create_folders, show_dataset_info

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🚀 FALL DETECTION SYSTEM - PROJECT SETUP")
    print("="*60 + "\n")
    
    # Create all necessary folders
    create_folders()
    
    # Show dataset information
    show_dataset_info()
    
    print("\n✅ Setup complete!")
    print("\n📋 NEXT STEPS:")
    print("1. Download the datasets using the URLs shown above")
    print("2. Extract them to the datasets/raw/ folders")
    print("3. Run the preprocessing scripts")
    print("4. Train your models with real data")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()