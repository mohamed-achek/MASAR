from pdf2image import convert_from_path

# Convert all pages
pages = convert_from_path("/home/codepips/Home/Portfolio/Projects/مسار/data/raw/Universities/TBS/TBS_Handbook-2022.pdf", dpi=300)

# Save each page as an image
for i, page in enumerate(pages):
    page.save(f"/home/codepips/Home/Portfolio/Projects/مسار/data/processed/tbs/page_{i+1}.png", "PNG")
