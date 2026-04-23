import tarfile
import numpy as np
import os
import io

snapshot_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/vector_raw-5233268114352168-2026-04-22-17-33-42.snapshot'
output_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/qwen_768_raw.npy'
DIM = 768
TARGET_COUNT = 28378

def extract():
    all_vectors = []
    print(f"Reading snapshot: {snapshot_path}")
    
    with tarfile.open(snapshot_path) as outer:
        segments = [n for n in outer.getnames() if 'segments/' in n and n.endswith('.tar')]
        
        for segment_name in segments:
            print(f"  Processing segment: {segment_name}")
            with outer.extractfile(segment_name) as inner_f:
                with tarfile.open(fileobj=inner_f) as inner:
                    # Look for vector data files
                    for member in inner.getmembers():
                        if 'matrix.dat' in member.name or 'chunk_0.mmap' in member.name:
                            print(f"    Extracting {member.name} ({member.size} bytes)")
                            with inner.extractfile(member) as f:
                                data = f.read()
                                # Convert to numpy f32
                                vec_array = np.frombuffer(data, dtype='<f4')
                                # Reshape (Count = Size / (DIM * 4))
                                num_vecs = len(vec_array) // DIM
                                vec_array = vec_array[:num_vecs * DIM].reshape(-1, DIM)
                                all_vectors.append(vec_array)
                                print(f"    Found {num_vecs} vectors.")

    if not all_vectors:
        print("No vectors found!")
        return

    # Concatenate all parts
    final_corpus = np.concatenate(all_vectors, axis=0)
    print(f"Total extracted from files: {final_corpus.shape}")

    # FILTER ZERO VECTORS
    print("Filtering zero vectors...")
    norms = np.linalg.norm(final_corpus, axis=1)
    valid_mask = norms > 1e-9
    final_corpus = final_corpus[valid_mask]
    print(f"Total non-zero vectors found: {final_corpus.shape[0]}")

    # Truncate to target count if requested
    if final_corpus.shape[0] > TARGET_COUNT:
        print(f"Truncating to {TARGET_COUNT}")
        final_corpus = final_corpus[:TARGET_COUNT]
    elif final_corpus.shape[0] < TARGET_COUNT:
        print(f"Warning: Only found {final_corpus.shape[0]} valid vectors, expected {TARGET_COUNT}")

    np.save(output_path, final_corpus)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    extract()
