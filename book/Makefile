LATEX = latex

DVIPS = dvips

PDFFLAGS = -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
           -dCompressPages=true -dUseFlateCompression=true  \
           -dEmbedAllFonts=true -dSubsetFonts=true -dMaxSubsetPct=100


%.dvi: %.tex
	$(LATEX) $<

%.ps: %.dvi
	$(DVIPS) -o $@ $<

%.pdf: %.ps
	ps2pdf $(PDFFLAGS) $<

all:	book.tex
	pdflatex $(PDFFLAGS) --shell-escape book
	makeindex book.idx
	evince book.pdf

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
DIST_FILES = ModSimPy.pdf

distrib:
	cp book.pdf ModSimPy.pdf
	rsync -a $(DIST_FILES) $(DEST)
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



