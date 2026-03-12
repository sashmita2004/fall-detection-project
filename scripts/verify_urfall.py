from pathlib import Path

def verify_urfall():
    """Verify UR Fall dataset extraction"""
    
    print("\n" + "="*70)
    print("✓ VERIFYING UR FALL DATASET")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    urfall_path = project_root / "datasets" / "raw" / "urfall"
    
    if not urfall_path.exists():
        print(f"\n❌ UR Fall folder not found: {urfall_path}")
        return False
    
    print(f"\n📂 Dataset location: {urfall_path}")
    
    # Count folders
    all_items = list(urfall_path.iterdir())
    folders = [item for item in all_items if item.is_dir()]
    files = [item for item in all_items if item.is_file()]
    
    fall_folders = [f for f in folders if 'fall-' in f.name.lower() and 'adl' not in f.name.lower()]
    adl_folders = [f for f in folders if 'adl-' in f.name.lower()]
    
    print(f"\n📊 Contents:")
    print(f"   Total folders: {len(folders)}")
    print(f"   Total files: {len(files)}")
    print(f"   Fall folders: {len(fall_folders)}")
    print(f"   ADL folders: {len(adl_folders)}")
    
    if len(fall_folders) > 0 or len(adl_folders) > 0:
        print(f"\n✅ Dataset looks good!")
        print(f"\nSample fall folders:")
        for f in fall_folders[:5]:
            num_images = len(list(f.glob('*.png'))) + len(list(f.glob('*.jpg'))) + len(list(f.glob('*.bmp')))
            print(f"   • {f.name} ({num_images} images)")
        
        if len(fall_folders) > 5:
            print(f"   ... and {len(fall_folders)-5} more")
        
        print(f"\nSample ADL folders:")
        for f in adl_folders[:5]:
            num_images = len(list(f.glob('*.png'))) + len(list(f.glob('*.jpg'))) + len(list(f.glob('*.bmp')))
            print(f"   • {f.name} ({num_images} images)")
        
        if len(adl_folders) > 5:
            print(f"   ... and {len(adl_folders)-5} more")
        
        print("\n✅ Ready for preprocessing!")
        return True
    else:
        print(f"\n⚠️ No fall/adl folders found!")
        print(f"\nCurrent contents:")
        for item in all_items[:20]:
            print(f"   • {item.name}")
        
        if len(all_items) > 20:
            print(f"   ... and {len(all_items)-20} more items")
        
        print("\n💡 If you see a single folder, the contents might be nested.")
        print("   Please move all fall-* and adl-* folders directly into urfall/")
        return False

if __name__ == "__main__":
    verify_urfall()