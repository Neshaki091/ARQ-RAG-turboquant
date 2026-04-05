import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');
import { createClient } from '@supabase/supabase-js';
import { embedText, generateChunkId } from '../services/embedding.js';

function supabase() {
  return createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    { auth: { persistSession: false } }
  );
}

const router = Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Đường dẫn tương đối từ `server/routes/ingest.js` ra thư mục gốc dự án
const METADATA_DIR = path.join(__dirname, '../../crawl-paper/document/metadata');

const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 200;

function chunkText(text) {
  const chunks = [];
  let index = 0;
  while (index < text.length) {
    chunks.push(text.substring(index, index + CHUNK_SIZE));
    index += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

const delay = (ms) => new Promise(r => setTimeout(r, ms));

router.get('/ingest-crawled', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders && res.flushHeaders();

  const emit = (type, data) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const db = supabase();
    
    // Check if truncate is true
    if (req.query.truncate === 'true') {
        emit('log', { message: 'Đang xóa dữ liệu cũ trong bảng documents...' });
        const { error } = await db.from('documents').delete().neq('id', 0);
        if (error) {
            emit('error', { message: 'Lỗi xóa DB: ' + error.message });
        } else {
            emit('log', { message: 'Xóa dữ liệu thành công.' });
        }
    }
    
    let txtFiles = [];
    try {
        const files = await fs.readdir(METADATA_DIR);
        txtFiles = files.filter(f => f.endsWith('.txt'));
    } catch (err) {
        emit('error', { message: 'Thư mục metadata chưa tồn tại hoặc bị lỗi đọc.' });
        res.end();
        return;
    }

    if (txtFiles.length === 0) {
        emit('log', { message: 'Không tìm thấy file nào trong thư mục crawl-paper/document/metadata' });
        emit('done', { status: 'complete' });
        res.end();
        return;
    }

    emit('stats', { total: txtFiles.length, current: 0 });

    let current = 0;
    for (const file of txtFiles) {
        current++;
        emit('stats', { total: txtFiles.length, current });
        
        const metadataPath = path.join(METADATA_DIR, file);
        const metadataContent = await fs.readFile(metadataPath, 'utf8');
        
        const lines = metadataContent.split('\n');
        const titleLine = lines.find(l => l.startsWith('Tiêu đề bài báo:') || l.startsWith('Tiêu đề:'));
        const authorLine = lines.find(l => l.startsWith('Tác giả:'));
        const linkLine = lines.find(l => l.startsWith('Link arXiv:') || l.startsWith('Link gốc:'));
        
        const heading = titleLine ? titleLine.replace(/Tiêu đề( bài báo)?:/, '').trim() : file;
        const authors = authorLine ? authorLine.replace('Tác giả:', '').trim() : '';
        const source = linkLine ? linkLine.replace(/Link( arXiv| gốc)?:/, '').trim() : 'Crawled paper';

        const pdfFilename = file.replace('.txt', '.pdf');

        let fullText = '';
        try {
            emit('log', { message: `📥 Đang tải PDF từ Cloud: ${pdfFilename}...` });
            const { data, error: downloadError } = await db.storage.from('papers').download(pdfFilename);
            if (downloadError) throw downloadError;
            
            const pdfBuffer = Buffer.from(await data.arrayBuffer());
            const pdfData = await pdfParse(pdfBuffer);
            fullText = pdfData.text;
            emit('log', { message: `✅ Đã trích xuất chữ thành công từ Cloud PDF: ${pdfFilename}` });
        } catch (err) {
            emit('log', { message: `⚠ Lỗi đọc PDF của ${file}: ${err.message}. Chuyển sang dùng tóm tắt.` });
            const summaryIndex = metadataContent.indexOf('--- TÓM TẮT (ABSTRACT) ---');
            if (summaryIndex !== -1) {
                fullText = metadataContent.substring(summaryIndex + 26).trim();
            } else {
                fullText = metadataContent;
            }
        }

        fullText = fullText.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
        if (!fullText) {
            emit('log', { message: `⚠ Bỏ qua bài báo rỗng.`});
            continue;
        }

        const chunks = chunkText(fullText);
        emit('log', { message: `🔄 Đang băm nhỏ [${heading.substring(0,35)}...] thành ${chunks.length} chunks.` });

        for (let i = 0; i < chunks.length; i++) {
           const chunkContent = chunks[i];
           const chunkId = generateChunkId(`${heading}-${authors}-${i}-${chunkContent}`);
           
           let success = false;
           // Retry mechanism for API limits
           for (let retry = 0; retry < 3; retry++) {
               try {
                   const embedding = await embedText(chunkContent);
                   
                   const { error } = await db.from('documents').upsert({
                       chunk_id: chunkId,
                       content: chunkContent,
                       heading: heading,
                       source: source,
                       chunk_index: i,
                       embedding: embedding
                   }, { onConflict: 'chunk_id', ignoreDuplicates: true });

                   if (error) throw error;
                   
                   emit('log', { message: `   → Đã nhúng & đối chiếu chunk ${i+1}/${chunks.length}` });
                   
                   // Delay to respect Gemini API limits (15 RPM)
                   await delay(4000); 
                   success = true;
                   break;
               } catch (err) {
                   emit('log', { message: `   ⚠ Lỗi nhúng chunk ${i+1}/${chunks.length}: ${err.message}. Đang thử lại (Retry ${retry+1}/3) sau 10s...` });
                   await delay(10000);
               }
           }
           
           if (!success) {
               emit('log', { message: `   ❌ Thất bại chunk ${i+1}/${chunks.length} sau 3 lần thử. Bỏ qua chunk này.` });
           }
        }
        
    }

    emit('log', { message: `🎉 Quá trình nhúng tất cả bài báo CRAWL đã hoàn tất!` });
    emit('done', { status: 'complete' });

  } catch (err) {
    emit('error', { message: err.message });
  } finally {
    res.end();
  }
});

export default router;
