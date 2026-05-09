import sqlite3
import os
import json

class MetadataService:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Lấy đường dẫn tuyệt đối đến thư mục data của backend
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "data", "metadata.db")
        else:
            self.db_path = db_path
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        print(f"🗄️ Metadata DB Path: {self.db_path}")
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    source TEXT,
                    page INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
            conn.commit()

    def add_chunks(self, start_id: int, chunks: list[dict]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = []
            for i, chunk in enumerate(chunks):
                data.append((
                    start_id + i,
                    chunk.get('text', ''),
                    chunk.get('source', ''),
                    chunk.get('page', 0)
                ))
            
            cursor.executemany(
                "INSERT OR REPLACE INTO chunks (id, text, source, page) VALUES (?, ?, ?, ?)",
                data
            )
            conn.commit()

    def get_by_ids(self, ids: list[int]):
        if not ids:
            return []
            
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Create placeholders for the IN clause
            placeholders = ','.join(['?'] * len(ids))
            query = f"SELECT * FROM chunks WHERE id IN ({placeholders})"
            
            cursor.execute(query, ids)
            rows = cursor.fetchall()
            
            # To match the expected format (MockPoint with payload attribute)
            class MockPoint:
                def __init__(self, payload):
                    self.payload = payload
            
            # Map back to original order of IDs if possible, or just return results
            results_dict = {row['id']: dict(row) for row in rows}
            
            final_results = []
            for idx in ids:
                if idx in results_dict:
                    final_results.append(MockPoint(results_dict[idx]))
            
            return final_results

    def delete_by_filename(self, filename: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get IDs before deleting to return to TQ for soft-delete
            cursor.execute("SELECT id FROM chunks WHERE source = ?", (filename,))
            ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("DELETE FROM chunks WHERE source = ?", (filename,))
            conn.commit()
            return ids

    def list_documents(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT source FROM chunks")
            return [row[0] for row in cursor.fetchall()]

    def get_count(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
            
    def reload(self):
        # No-op for SQLite as it always reads from disk
        pass

# Global instance
metadata_service = MetadataService()
