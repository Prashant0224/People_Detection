import pdfkit

# Give the full path to wkhtmltopdf.exe
path_to_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"

# Tell pdfkit where to find wkhtmltopdf
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

# Convert resume.html into a PDF file
pdfkit.from_file("resume.html", "Prashant_Litoriya_Resume.pdf", configuration=config)

print("✅ Resume PDF generated successfully!")
