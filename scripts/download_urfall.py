import webbrowser
from pathlib import Path
import os

def download_urfall_guide():
    """Guide for downloading UR Fall Detection dataset"""
    
    print("\n" + "="*70)
    print("📥 UR FALL DETECTION DATASET DOWNLOAD GUIDE")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    urfall_path = project_root / "datasets" / "raw" / "urfall"
    
    print(f"\n📍 Extract files to: {urfall_path}")
    
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    
    print("\n🌐 OPTION 1: Official Source (May require academic email)")
    print("   URL: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html")
    print("   Size: ~1.8 GB")
    print("   Quality: Best, most complete")
    
    print("\n🌐 OPTION 2: Kaggle (Recommended for quick access)")
    print("   Steps:")
    print("   1. Go to: https://www.kaggle.com")
    print("   2. Search: 'UR Fall Detection' or 'Fall Detection RGB-D'")
    print("   3. Create free account if needed")
    print("   4. Click 'Download' button")
    print("   5. Extract ZIP file")
    
    print("\n🌐 OPTION 3: Alternative Repository")
    print("   URL: https://github.com/Rumeysakeskin/Fall-Detection")
    print("   (May have sample data or links)")
    
    print("\n" + "="*70)
    print("WHAT TO EXPECT IN UR FALL DATASET")
    print("="*70)
    print("\n📁 Folder structure after extraction:")
    print("   urfall/")
    print("   ├── fall-01/")
    print("   │   ├── rgb/           (video frames)")
    print("   │   ├── depth/         (depth images)")
    print("   │   └── accelerometer/ (sensor data)")
    print("   ├── fall-02/")
    print("   ├── ...")
    print("   ├── adl-01/")
    print("   ├── adl-02/")
    print("   └── ...")
    
    print("\n" + "="*70)
    print("STEP-BY-STEP INSTRUCTIONS")
    print("="*70)
    print("\n1️⃣  I'll open the download pages for you")
    print("2️⃣  Choose the easiest option (usually Kaggle)")
    print("3️⃣  Download the ZIP file (1.8 GB)")
    print("4️⃣  Extract to your Downloads folder")
    print("5️⃣  Copy ALL folders to the urfall path shown above")
    print("6️⃣  Come back here and type: 'UR Fall downloaded'")
    
    print("\n⏰ Expected download time: 15-30 minutes")
    print("💾 Required space: 5 GB (after extraction)")
    
    # Offer to open browsers
    print("\n" + "="*70)
    choice = input("❓ Open Kaggle (easiest option)? (y/n): ")
    if choice.lower() == 'y':
        print("🌐 Opening Kaggle...")
        webbrowser.open('https://www.kaggle.com/datasets')
        print("✓ Search for: 'Fall Detection' or 'UR Fall'")
    
    choice = input("\n❓ Open official UR Fall page? (y/n): ")
    if choice.lower() == 'y':
        print("🌐 Opening UR Fall official page...")
        webbrowser.open('http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html')
        print("✓ Look for download link")
    
    choice = input("\n❓ Open GitHub alternative? (y/n): ")
    if choice.lower() == 'y':
        print("🌐 Opening GitHub...")
        webbrowser.open('https://github.com/search?q=fall+detection+dataset&type=repositories')
        print("✓ Look for UR Fall or similar datasets")
    
    print("\n" + "="*70)
    print("💡 ALTERNATIVE: Use Sample Data (For Testing)")
    print("="*70)
    print("\nIf download is too slow, I can help you:")
    print("• Create synthetic video data for testing")
    print("• Use a smaller subset")
    print("• Skip video models and use sensor-only")
    
    print("\n📝 After downloading, tell me:")
    print("   'UR Fall downloaded' - if successful")
    print("   'Download is slow' - if having issues")
    print("   'Use synthetic data' - if you want to skip real video data")
    print("\n")

if __name__ == "__main__":
    download_urfall_guide()