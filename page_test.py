from backend.main import extract_pages_from_pdf

PDF_PATH = "backend/uploads/EduLex_SE_Lab_Report.pdf"

pages = extract_pages_from_pdf(PDF_PATH)

print("Number of pages:", len(pages))

for page in pages:
    print("\n" + "=" * 80)
    print("PAGE:", page["page_number"])
    print("Characters:", len(page["text"]))
    print("First 200 characters:")
    print(page["text"][:200])