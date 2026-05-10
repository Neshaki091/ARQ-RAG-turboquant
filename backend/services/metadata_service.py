import sqlite3
import os
import json

class MetadataService:
    def __init__(self, db_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.system_db_path = os.path.join(self.data_dir, "metadata_system.db")
        self.user_db_path = os.path.join(self.data_dir, "metadata_user.db")
        
        print(f"DB: System Path: {self.system_db_path}")
        print(f"DB: User Path: {self.user_db_path}")
        self._init_db()

    def _get_connection(self, user_id: int = -1):
        # Định tuyến Database dựa trên user_id
        db_path = self.system_db_path if user_id == -1 else self.user_db_path
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self):
        # 1. Khởi tạo System DB
        with self._get_connection(user_id=-1) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    source TEXT,
                    page INTEGER,
                    user_id INTEGER,
                    session_id TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sys_session ON chunks(session_id)')

        # 2. Khởi tạo User DB
        with self._get_connection(user_id=0) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    source TEXT,
                    page INTEGER,
                    user_id INTEGER,
                    session_id TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_session ON chunks(user_id, session_id)')

    def add_user(self, username, password, role='user'):
        try:
            with self._get_connection(user_id=0) as conn: # Luôn vào User DB
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
                conn.commit()
                return cursor.lastrowid
        except Exception:
            return None

    def get_user(self, username):
        with self._get_connection(user_id=0) as conn: # Luôn vào User DB
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            return cursor.fetchone()

    def add_chunks(self, start_id: int, chunks: list[dict], user_id: int = -1, session_id: str = "default"):
        with self._get_connection(user_id=user_id) as conn:
            cursor = conn.cursor()
            data = []
            for i, chunk in enumerate(chunks):
                data.append((
                    start_id + i,
                    chunk.get('text', ''),
                    chunk.get('source', ''),
                    chunk.get('page', 0),
                    user_id,
                    session_id
                ))
            
            cursor.executemany(
                "INSERT OR REPLACE INTO chunks (id, text, source, page, user_id, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                data
            )
            conn.commit()

    def get_by_ids(self, ids: list[int], user_id: int = None, session_id: str = "default", scope: str = "both"):
        """
        Lấy metadata từ 2 DB khác nhau dựa trên scope
        """
        if not ids: return []
        
        results_dict = {}
        placeholders = ','.join(['?'] * len(ids))

        # 1. Query System DB (Wikipedia)
        if scope in ["system", "both"]:
            with self._get_connection(user_id=-1) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND user_id = -1"
                cursor.execute(query, ids)
                for row in cursor.fetchall():
                    results_dict[row['id']] = dict(row)

        # 2. Query User DB (Cá nhân)
        if scope in ["user", "both"] and user_id is not None:
            with self._get_connection(user_id=user_id) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND user_id = ? AND session_id = ?"
                cursor.execute(query, ids + [user_id, session_id])
                for row in cursor.fetchall():
                    results_dict[row['id']] = dict(row)

        # 3. Gộp kết quả theo thứ tự ID truyền vào
        class MockPoint:
            def __init__(self, payload):
                self.payload = payload

        final_results = []
        for idx in ids:
            if idx in results_dict:
                final_results.append(MockPoint(results_dict[idx]))
        
        return final_results

    def delete_by_filename(self, filename: str, user_id: int):
        with self._get_connection(user_id=user_id) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM chunks WHERE source = ? AND user_id = ?", (filename, user_id))
            ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("DELETE FROM chunks WHERE source = ? AND user_id = ?", (filename, user_id))
            conn.commit()
            return ids

    def list_documents(self, user_id: int = -1):
        with self._get_connection(user_id=user_id) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT source FROM chunks WHERE user_id = ?", (user_id,))
            return [row[0] for row in cursor.fetchall()]

    def get_count(self, user_id: int = -1):
        with self._get_connection(user_id=user_id) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            res = cursor.fetchone()
            return res[0] if res else 0
            
    def reload(self):
        pass

    def get_ids_by_session(self, user_id: int, session_id: str):
        with self._get_connection(user_id=user_id) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM chunks WHERE user_id = ? AND session_id = ?", (user_id, session_id))
            return [row[0] for row in cursor.fetchall()]

    def get_chunk(self, chunk_id: int, user_id: int = -1):
        with self._get_connection(user_id=user_id) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_chunks(self, user_id: int = -1, limit: int = 100, offset: int = 0):
        with self._get_connection(user_id=user_id) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Lấy 100 đoạn văn (window 100)
            cursor.execute("SELECT * FROM chunks LIMIT ? OFFSET ?", (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def list_all_users(self):
        with self._get_connection(user_id=0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, role, created_at FROM users")
            return [dict(row) for row in cursor.fetchall()]

    def delete_by_filename(self, filename: str, user_id: int):
        with self._get_connection(user_id=user_id) as conn:
            cursor = conn.cursor()
            # Lấy danh sách ID để xóa vector bên TQ sau
            cursor.execute("SELECT id FROM chunks WHERE source = ? AND user_id = ?", (filename, user_id))
            ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("DELETE FROM chunks WHERE source = ? AND user_id = ?", (filename, user_id))
            conn.commit()
            return ids

    def get_chunks_by_filename(self, filename: str, user_id: int):
        with self._get_connection(user_id=user_id) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE source = ? AND user_id = ? ORDER BY id", (filename, user_id))
            return [dict(row) for row in cursor.fetchall()]

    def get_chunk_metadata(self, chunk_id: int, user_id: int = -1):
        # Alias cho get_chunk để tương thích ngược
        return self.get_chunk(chunk_id, user_id)

# Global instance
metadata_service = MetadataService()

