LATEX = latex

DVIPS = dvips

PDFFLAGS = -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
           -dEmbedAllFonts=true -dSubsetFonts=true -dMaxSubsetPct=100
#           -dCompressPages=true -dUseFlateCompression=true  \


%.dvi: %.tex
	$(LATEX) $<

%.ps: %.dvi
	$(DVIPS) -o $@ $<

%.pdf: %.ps
	ps2pdf $<

# this book uses siunitx, which you can install with apt
# sudo apt-get install texlive-science 

all:	book.tex
	pdflatex --shell-escape book
	makeindex book.idx
	evince book.pdf

ps:
	pdf2ps book.pdf
	# and then upload to https://www.ps2pdf.com/convert-ps-to-pdf

hevea:
	#sed 's/\(figs\/[^.]*\).\(pdf\|png\)/\1.eps/' 
	cp book.tex ModSimPy.tex
	rm -rf html
	mkdir html
	hevea -O -e latexonly htmlonly ModSimPy
# the following greps are a kludge to prevent imagen from seeing
# the definitions in latexonly, and to avoid headers on the images
	grep -v latexonly ModSimPy.image.tex > a; mv a ModSimPy.image.tex
	grep -v fancyhdr ModSimPy.image.tex > a; mv a ModSimPy.image.tex
	imagen -png ModSimPy
	hacha ModSimPy.html
	cp up.png next.png back.png html
	mv index.html ModSimPy.css ModSimPy*.html  html
	mv ModSimPy*.png *motif.gif html

DEST = /home/downey/public_html/greent/ModSimPy/
PDF = ModSimPy3.pdf

distrib:
	cp book.pdf $(PDF)
	rsync -a $(PDF) $(DEST)
	#chmod -R o+r $(DEST)/*
	cd /home/downey/public_html/greent; sh back

plastex:
	# Before running plastex, we need the current directory in PYTHONPATH
	# export PYTHONPATH=$PYTHONPATH:.
	# rm -rf /home/downey/ModSimPy/book/.svn
	plastex --renderer=DocBook --theme=book --image-resolution=300 --filename=book.xml book.tex
	# rm -rf /home/downey/ModSimPy/trunk/book/.svn

xxe:
	xmlcopyeditor ~/ModSimPy/book/book/book.xml &
	#~/Downloads/xxe-perso-4_8_0/bin/xxe book/book.xml

lint:
	xmllint -noout book/book.xml

OREILLY = atlas

oreilly:
	rsync -a book/ $(OREILLY)
	rsync -a figs/* $(OREILLY)/figs/
	cd $(OREILLY); git add .
	cd $(OREILLY); git commit -m "Automated check in."
	cd $(OREILLY); git push

clean:
	rm -f *~ *.aux *.log *.dvi *.idx *.ilg *.ind *.toc

split:
	cp book.pdf ModSimPy.pdf
	rm -f ModSimPyChapters/*
	~/Downloads/sejda-console-3.2.51/bin/sejda-console splitbybookmarks -f ModSimPy.pdf -l 1 -o ModSimPyChapters


