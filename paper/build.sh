INPUT_TXT="paper.md"

INPUT_BIB="paper.bib"

OUTPUT_PDF="paper.pdf"

ENGINE="xelatex"

OPTS="-V geometry:margin=1in"

pandoc --citeproc ${OPTS} --bibliography=${INPUT_BIB} --pdf-engine=${ENGINE} -s ${INPUT_TXT} -o ${OUTPUT_PDF}


