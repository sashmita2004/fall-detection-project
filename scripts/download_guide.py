import webbrowser
from pathlib import Path

def show_download_instructions():
    """Shows step-by-step download instructions"""
    
    print("\n" + "="*70)
    print("📥 DATASET DOWNLOAD GUIDE")
    print("="*70)
    
    # Get project paths
    project_root = Path(__file__).parent.parent
    urfall_path = project_root / "datasets" / "raw" / "urfall"
    sisfall_path = project_root / "datasets" / "raw" / "sisfall"
    
    print("\n📍 Your dataset folders are ready at:")
    print(f"   UR Fall: {urfall_path}")
    print(f"   SisFall: {sisfall_path}")
    
    print("\n" + "="*70)
    print("OPTION 1: UR FALL DETECTION DATASET (1.8 GB)")
    print("="*70)
    print("\n📋 Steps:")
    print("1. I'll open the download page in your browser")
    print("2. Look for 'Download' or 'Dataset' link on the page")
    print("3. Download the ZIP file to your Downloads folder")
    print("4. Extract the ZIP file")
    print(f"5. Copy ALL folders to: {urfall_path}")
    
    choice = input("\n❓ Open UR Fall download page? (y/n): ")
    if choice.lower() == 'y':
        print("🌐 Opening browser...")
        webbrowser.open('http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html')
        print("✓ Browser opened!")
    
    print("\n" + "="*70)
    print("OPTION 2: SISFALL DATASET (350 MB) - EASIER TO START WITH")
    print("="*70)
    print("\n📋 Steps:")
    print("1. I'll open the download page in your browser")
    print("2. Click on 'Download SisFall Dataset' button")
    print("3. Save the ZIP file to your Downloads folder")
    print("4. Extract the ZIP file")
    print(f"5. Copy ALL folders (SA01, SA02, etc.) to: {sisfall_path}")
    
    choice = input("\n❓ Open SisFall download page? (y/n): ")
    if choice.lower() == 'y':
        print("🌐 Opening browser...")
        webbrowser.open('http://sistemic.udea.edu.co/en/research/projects/english-falls/')
        print("✓ Browser opened!")
    
    print("\n" + "="*70)
    print("ALTERNATIVE: KAGGLE (EASIER FOR BEGINNERS)")
    print("="*70)
    print("\n📋 If the above sites don't work, try Kaggle:")
    print("1. Create free account at: https://www.kaggle.com")
    print("2. Search for: 'SisFall dataset' or 'Fall detection dataset'")
    print("3. Click 'Download' button")
    print("4. Extract and copy to the appropriate folder")
    
    choice = input("\n❓ Open Kaggle? (y/n): ")
    if choice.lower() == 'y':
        print("🌐 Opening browser...")
        webbrowser.open('https://www.kaggle.com/datasets')
        print("✓ Browser opened!")
    
    print("\n" + "="*70)
    print("⚠️ IMPORTANT NOTES:")
    print("="*70)
    print("• Download SisFall FIRST (smaller, easier)")
    print("• Make sure you have 5 GB free disk space")
    print("• Downloads might take 30-60 minutes depending on your internet")
    print("• Don't extract inside another folder - copy directly to raw/")
    print("\n💡 TIP: Start with SisFall, it's smaller and faster to download!")
    print("="*70)
    
    print("\n📝 After downloading, tell me:")
    print("   'Downloaded SisFall' or 'Downloaded UR Fall' or 'Need help'")
    print("\n")

if __name__ == "__main__":
    show_download_instructions()