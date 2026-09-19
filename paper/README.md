# Paper build notes

The manuscript is `paper.tex`. It uses the IEEE conference class and BibTeX
style already included in this directory. `latexmkrc` adds those two template
folders to TeX's search path.

From this directory, build with:

```bash
latexmk -pdf paper.tex
```

If `latexmk` is unavailable but a full TeX distribution is installed, run:

```bash
TEXINPUTS="$PWD/IEEE-conference-template-062824 2:" \
BSTINPUTS="$PWD/IEEEtranBST2 (1):" \
pdflatex paper.tex
TEXINPUTS="$PWD/IEEE-conference-template-062824 2:" \
BSTINPUTS="$PWD/IEEEtranBST2 (1):" \
bibtex paper
TEXINPUTS="$PWD/IEEE-conference-template-062824 2:" \
BSTINPUTS="$PWD/IEEEtranBST2 (1):" \
pdflatex paper.tex
TEXINPUTS="$PWD/IEEE-conference-template-062824 2:" \
BSTINPUTS="$PWD/IEEEtranBST2 (1):" \
pdflatex paper.tex
```

Regenerate the result overview figure with:

```bash
/venv/tall-5060/bin/python make_figures.py
```

Before submission, replace the anonymous author block, choose the venue's
required page limit, and decide whether the appendices belong in the main PDF
or a supplement. The current machine does not have a TeX engine installed, so
the source was structurally checked but not compiled here.
