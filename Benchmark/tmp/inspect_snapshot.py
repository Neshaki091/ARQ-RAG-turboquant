import tarfile
import os

snapshot_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/vector_raw-5233268114352168-2026-04-22-17-33-42.snapshot'

def inspect_snapshot():
    print(f"--- Inspecting Outer Tar: {snapshot_path} ---")
    if not os.path.exists(snapshot_path):
        print("File not found!")
        return

    with tarfile.open(snapshot_path) as outer:
        all_names = outer.getnames()
        print(f"Total files in outer tar: {len(all_names)}")
        for name in all_names[:20]:
            print(f"  {name}")
        
        # Check for segment files
        segments = [n for n in all_names if 'segments/' in n and n.endswith('.tar')]
        print(f"\nFound {len(segments)} segment files.")
        
        if segments:
            # Inspect specifically for vector/metadata files
            for target in segments[:2]: # Check first two segments
                print(f"\n--- Inspecting Inner Tar: {target} ---")
                inner_f = outer.extractfile(target)
                if inner_f is not None:
                    with tarfile.open(fileobj=inner_f) as inner:
                        for member in inner.getmembers():
                            # Print names and sizes
                            print(f"    {member.name:<60} | {member.size:>10} bytes")
                else:
                    print(f"    Could not extract {target}")

if __name__ == "__main__":
    inspect_snapshot()
