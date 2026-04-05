import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load biến môi trường từ thư mục RAG gốc
dotenv.config({ path: path.join(__dirname, '../.env') });

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!supabaseUrl || !supabaseKey) {
  console.error("❌ Không tìm thấy SUPABASE_URL hoặc SUPABASE_SERVICE_ROLE_KEY trong file .env!");
  process.exit(1);
}
const supabase = createClient(supabaseUrl, supabaseKey);
const OUTPUT_DIR = path.join(__dirname, 'document');
const METADATA_DIR = path.join(OUTPUT_DIR, 'metadata');

// Cấu hình tìm kiếm
const TARGET_TOTAL = 100; // Đã giảm xuống 100 để tránh bị ban IP khi tải PDF
const QUERIES = [
  { name: 'TurboQuant', query: 'all:"TurboQuant" OR all:"PolarQuant" OR all:"Quantized Johnson-Lindenstrauss"' },
  { name: 'ARQ', query: 'all:"Asynchronous RAG" OR all:"ARQ"' },
  { name: 'PQ', query: 'all:"Product Quantization" OR all:"Vector Quantization"' },
  { name: 'RAG', query: 'all:"Retrieval-Augmented Generation" OR all:"RAG"' }
];

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function cleanFilename(str) {
  return str.replace(/[\/\\?%*:|"<>]/g, '-').trim();
}

async function extractID(url) {
  const parts = url.split('/');
  return parts[parts.length - 1] || 'unknown';
}

async function crawlArxivWithPDF() {
  console.log(`🔍 Bắt đầu cào ~${TARGET_TOTAL} bài báo từ arXiv (KÈM THEO FILE PDF)...\n`);
  console.log(`⚠️ LƯU Ý: Quá trình sẽ chậm hơn đáng kể do cần tải trọn bộ file PDF (vài MB/file).`);
  console.log(`⚠️ arXiv có chính sách chống spam, script sẽ delay 2s mỗi bài báo để tránh bị ban IP.\n`);
  
  await fs.mkdir(METADATA_DIR, { recursive: true });
  console.log(`📁 Thư mục lưu trữ metadata: ${METADATA_DIR} \n   (PDF sẽ được tải thẳng lên hệ thống Supabase Storage)\n`);

  let totalSaved = 0;
  const savedIds = new Set();
  const maxPerQuery = 50; // Tổng 4 chủ đề x 50 bài = 200 bài

  for (const q of QUERIES) {
    if (totalSaved >= TARGET_TOTAL) break;

    console.log(`\n⏳ Đang tìm kiếm chủ đề: [${q.name}]...`);
    const encoded = encodeURIComponent(q.query);
    const apiUrl = `http://export.arxiv.org/api/query?search_query=${encoded}&start=0&max_results=${maxPerQuery}&sortBy=relevance&sortOrder=descending`;
    
    try {
      const res = await fetch(apiUrl);
      const xml = await res.text();
      const entries = xml.split('<entry>').slice(1);
      
      let countForQuery = 0;

      for (const entry of entries) {
        if (totalSaved >= TARGET_TOTAL) break;

        const titleMatch = entry.match(/<title>([\s\S]*?)<\/title>/);
        const summaryMatch = entry.match(/<summary>([\s\S]*?)<\/summary>/);
        const linkMatch = entry.match(/<id>([\s\S]*?)<\/id>/);
        const publishedMatch = entry.match(/<published>([\s\S]*?)<\/published>/);
        const pdfUrlMatch = entry.match(/<link title="pdf" href="([^"]+)"/);

        // Trích xuất tác giả
        const authors = [];
        const authorRegex = /<author>\s*<name>([\s\S]*?)<\/name>\s*<\/author>/g;
        let match;
        while ((match = authorRegex.exec(entry)) !== null) {
          authors.push(match[1].trim());
        }

        const link = linkMatch ? linkMatch[1].trim() : 'unknown';
        const id = await extractID(link);
        const pdfUrl = pdfUrlMatch ? pdfUrlMatch[1].replace('http://', 'https://') + '.pdf' : `https://arxiv.org/pdf/${id}.pdf`;
        
        if (savedIds.has(id)) continue;
        savedIds.add(id);

        const title = titleMatch ? titleMatch[1].replace(/\n/g, ' ').replace(/\s+/g, ' ').trim() : 'Untitled';
        const summary = summaryMatch ? summaryMatch[1].replace(/\n/g, ' ').replace(/\s+/g, ' ').trim() : '';
        const published = publishedMatch ? publishedMatch[1].split('T')[0] : 'Unknown';
        const baseFilename = `${id}_${cleanFilename(title).substring(0, 50)}`;

        const pdfFilename = `${baseFilename}.pdf`;
        
        // Tạo biến Storage URL công khai (Mặc định trước theo Supabase)
        const { data: publicUrlData } = supabase.storage.from('papers').getPublicUrl(pdfFilename);
        const supabaseUrlLink = publicUrlData.publicUrl;

        const txtPath = path.join(METADATA_DIR, `${baseFilename}.txt`);

        // ==========================================
        // TẢI RAW PDF & UPLOAD CLOUD
        // ==========================================
        console.log(` 📥 Đang tải PDF: ${id} (${countForQuery + 1}) - ${title.substring(0, 30)}...`);
        try {
          const pdfRes = await fetch(pdfUrl);
          if (pdfRes.ok) {
            const pdfBuffer = Buffer.from(await pdfRes.arrayBuffer());
            // Upload Storage
            console.log(`   ⬆️ Đang Upload PDF gốc lên Supabase Storage (Bucket: papers)...`);
            const { error: uploadError } = await supabase.storage
              .from('papers')
              .upload(pdfFilename, pdfBuffer, {
                contentType: 'application/pdf',
                upsert: true
              });
              
            if (uploadError) {
              console.log(`   ❌ Lỗi Upload Supabase: ${uploadError.message}`);
            } else {
              console.log(`   ✅ Đã chép file lên Supabase thành công.`);
              
              // Chỉ lưu metadata file khi PDF đã upload thành công
              const fileContent = 
`Tiêu đề bài báo: ${title}
Tác giả: ${authors.join(', ')}
Ngày xuất bản: ${published}
Nền tảng: arXiv
Link gốc: ${link}
Link PDF: ${pdfUrl}
Link Lưu Trữ (Supabase): ${supabaseUrlLink}
Chủ đề Crawler: ${q.name}

--- TÓM TẮT (ABSTRACT) ---
${summary}
`;
              await fs.writeFile(txtPath, fileContent, 'utf8');
              console.log(`   ✅ Đã tạo file metadata thành công.`);
            }
          } else {
            console.log(`   ❌ Bỏ qua PDF này (lỗi HTTP ${pdfRes.status})`);
          }
        } catch (downloadErr) {
          console.log(`   ❌ Bỏ qua PDF này (lỗi mạng)`);
        }

        countForQuery++;
        totalSaved++;

        // Delay 2s để tránh bị block ban IP khi request file PDF liên tục (arXiv giới hạn rất gắt PDF crawler)
        await delay(2000);
      }

      console.log(`✅ Đã cào xong ${countForQuery} bài báo + PDF cho [${q.name}]. Tổng thư viện: ${totalSaved}/${TARGET_TOTAL}`);
    } catch (err) {
      console.error(`❌ Lỗi API khi tìm [${q.name}]:`, err.message);
    }
  }

  console.log(`\n🎉 HOÀN THÀNH TOÀN BỘ!`);
  console.log(`Đã tải về TẤT CẢ FILE .TXT và .PDF vào thư mục:`);
  console.log(`👉 ${OUTPUT_DIR}`);
}

crawlArxivWithPDF();
