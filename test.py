from PyPDF2 import PdfMerger

# List of PDF files to merge
pdf_files = ["1.pdf", "2.pdf", "3.pdf"]

# Create a PdfMerger object
merger = PdfMerger()

# Append each file to the merger
for pdf in pdf_files:
    merger.append(pdf)

# Output the merged PDF
merger.write("luat_dat_dai.pdf")
merger.close()

print("PDF files merged successfully!")
