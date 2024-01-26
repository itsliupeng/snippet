from pdf2docx import Converter

pdf_file = '/Users/liupeng/data/电子教材文本化20230913-7/原始PDF/人教版高中地理必修1.pdf'
docx_file = '/Users/liupeng/data/电子教材文本化20230913-7/人教版高中地理必修1.docx'

# convert pdf to docx
cv = Converter(pdf_file)
cv.convert(docx_file)      # all pages by default
cv.close()