import sys

def extract_pdf(pdf_path, output_path):
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text() or ""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Done PyPDF2")
    except ImportError:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += f"\n--- Page {page.number + 1} ---\n"
                text += page.get_text()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print("Done PyMuPDF")
        except ImportError:
            print("No PDF library found. Please install PyMuPDF (pip install pymupdf) or PyPDF2 (pip install pypdf2).")

if __name__ == "__main__":
    extract_pdf(sys.argv[1], sys.argv[2])
