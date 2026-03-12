import zipfile
import shutil
from pathlib import Path
import os

def find_zip_files():
    """Find potential UR Fall ZIP files"""
    
    print("\n" + "="*70)
    print("🔍 SEARCHING FOR UR FALL ZIP FILES")
    print("="*70)
    
    # Common download locations
    common_paths = [
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / "Documents"
    ]
    
    zip_files = []
    
    for path in common_paths:
        if path.exists():
            found = list(path.glob("*.zip"))
            if found:
                print(f"\n📁 Found in {path}:")
                for zf in found:
                    size_mb = zf.stat().st_size / (1024 * 1024)
                    print(f"   • {zf.name} ({size_mb:.1f} MB)")
                    zip_files.append(zf)
    
    return zip_files

def extract_urfall():
    """Extract UR Fall dataset"""
    
    print("\n" + "="*70)
    print("📦 UR FALL DATASET EXTRACTION TOOL")
    print("="*70)
    
    # Find ZIP files
    zip_files = find_zip_files()
    
    if not zip_files:
        print("\n❌ No ZIP files found in common locations")
        print("\n💡 Please manually locate the ZIP file and provide the path")
        zip_path = input("\nEnter full path to ZIP file (or 'skip' to exit): ")
        
        if zip_path.lower() == 'skip':
            return
        
        zip_path = Path(zip_path.strip().strip('"'))
        if not zip_path.exists():
            print(f"❌ File not found: {zip_path}")
            return
        zip_files = [zip_path]
    
    # Ask which file to extract
    if len(zip_files) > 1:
        print("\n📋 Select which file to extract:")
        for i, zf in enumerate(zip_files, 1):
            print(f"{i}. {zf.name}")
        
        choice = input("\nEnter number (or 'all' for all files): ")
        if choice.lower() == 'all':
            selected = zip_files
        else:
            try:
                selected = [zip_files[int(choice) - 1]]
            except:
                print("❌ Invalid choice")
                return
    else:
        selected = zip_files
    
    # Destination
    project_root = Path(__file__).parent.parent
    dest_path = project_root / "datasets" / "raw" / "urfall"
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Extracting to: {dest_path}")
    
    # Extract each file
    for zip_file in selected:
        print(f"\n📦 Extracting {zip_file.name}...")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of files
                file_list = zip_ref.namelist()
                print(f"   Contains {len(file_list)} files")
                
                # Extract with progress
                total = len(file_list)
                for i, file in enumerate(file_list, 1):
                    zip_ref.extract(file, dest_path)
                    if i % 100 == 0:
                        print(f"   Progress: {i}/{total} files ({i/total*100:.1f}%)")
                
                print(f"✓ Extraction complete!")
        
        except Exception as e:
            print(f"❌ Error extracting: {e}")
            continue
    
    # Check what was extracted
    print("\n📊 Checking extracted contents...")
    items = list(dest_path.iterdir())
    
    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    print(f"\n✓ Extracted:")
    print(f"   Folders: {len(folders)}")
    print(f"   Files: {len(files)}")
    
    if folders:
        print(f"\n📁 First few folders:")
        for folder in folders[:10]:
            print(f"   • {folder.name}")
    
    # If extracted into a subfolder, move contents up
    if len(folders) == 1 and len(files) == 0:
        subfolder = folders[0]
        print(f"\n💡 Detected nested structure in: {subfolder.name}")
        print("   Moving contents up one level...")
        
        for item in subfolder.iterdir():
            shutil.move(str(item), str(dest_path))
        
        subfolder.rmdir()
        print("✓ Moved contents up")
    
    print("\n" + "="*70)
    print("✅ EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\n📂 Dataset location: {dest_path}")
    print("\n💡 Next step:")
    print("   Run: python scripts/preprocess_urfall.py")
    print("\n")

if __name__ == "__main__":
    extract_urfall()