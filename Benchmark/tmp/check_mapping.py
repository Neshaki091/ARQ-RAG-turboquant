import tarfile
import struct
import io

snapshot_path = 'f:/IT project/DoAn/Turboquant-rust demo/turboquant_v2/data/vector_raw-5233268114352168-2026-04-22-17-33-42.snapshot'

def check_mapping():
    print(f"Opening snapshot for mapping: {snapshot_path}")
    with tarfile.open(snapshot_path) as outer:
        mapping_file = '0/segments/ee77a8ad-edf2-4ff7-add5-2b73f94412e3.tar'
        with outer.extractfile(mapping_file) as inner_f:
            with tarfile.open(fileobj=inner_f) as inner:
                m_f = inner.extractfile('snapshot/files/id_tracker.mappings')
                if m_f:
                    data = m_f.read()
                    print(f"Mapping file size: {len(data)} bytes")
                    # Qdrant internal IDs are often 8-byte integers or 16-byte UUIDs.
                    # Let's try 8-byte pairs (internal_id, external_id)
                    count = len(data) // 16
                    print(f"Estimated entries (16-byte): {count}")
                    for i in range(min(10, count)):
                        entry = data[i*16 : (i+1)*16]
                        # Try decoding as two 8-byte uint64
                        ids = struct.unpack('<QQ', entry)
                        print(f"  Entry {i}: Internal={ids[0]}, External={ids[1]}")

if __name__ == "__main__":
    check_mapping()
