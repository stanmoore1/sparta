# Build the SPARTA manual with Sphinx.
#
# Sphinx and its dependencies are installed into a virtualenv under
# doc/docenv the first time a build runs, so the only prerequisite is a
# python3 with venv available.

SHELL          = /bin/bash
BUILDDIR       = $(shell pwd)
RSTDIR         = $(BUILDDIR)/src
VENV           = $(BUILDDIR)/docenv
SPHINXCONFIG   = $(BUILDDIR)/utils/sphinx-config
MATHJAX        = $(SPHINXCONFIG)/_static/mathjax
MATHJAXVER     = 3.2.2
MATHJAXURL     = https://registry.npmjs.org/mathjax/-/mathjax-$(MATHJAXVER).tgz
MATHJAXSHA     = 1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09
PYTHON         = python3
SPHINXEXTRA    =

.PHONY: help clean clean-all html pdf fasthtml

HAS_PDFLATEX   = $(shell command -v latexmk > /dev/null 2>&1 && \
                         command -v pdflatex > /dev/null 2>&1 && echo YES || echo NO)
HAS_PANDOC     = $(shell command -v pandoc > /dev/null 2>&1 && echo YES || echo NO)
FASTHTMLFILES  = $(patsubst $(RSTDIR)/%.rst,fasthtml/%.html,$(wildcard $(RSTDIR)/*.rst))

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  html          build the HTML manual in doc/html"
	@echo "  pdf           build doc/Manual.pdf (needs a LaTeX installation)"
	@echo "  fasthtml      quick per-page preview in doc/fasthtml (needs pandoc)"
	@echo "  clean         remove the build tree, keep the virtualenv"
	@echo "  clean-all     also remove the virtualenv and generated conf.py"
	@echo ""
	@echo "  SPHINXEXTRA=... passes extra flags to sphinx-build, e.g."
	@echo "  make html SPHINXEXTRA=\"-j auto -W --keep-going\""

clean:
	rm -rf $(BUILDDIR)/html $(BUILDDIR)/doctrees \
	       $(BUILDDIR)/latex $(BUILDDIR)/latexdoctrees $(BUILDDIR)/Manual.pdf \
	       $(BUILDDIR)/fasthtml

clean-all: clean
	rm -rf $(VENV) $(SPHINXCONFIG)/conf.py $(MATHJAX)

# the virtualenv is a directory target: it is built once, then only rebuilt
# if requirements.txt changes
#
# Chained with && rather than ; so a failure stops here.  With ; a failed
# "python3 -m venv" would go on to install Sphinx into whatever interpreter
# is on PATH, and a failed "pip install" would still reach the touch below,
# leaving a half-built virtualenv that make then considers up to date.
# A partial virtualenv is also removed on the way out: it would otherwise be
# left newer than requirements.txt, so the next make would take it for a
# finished one and build the manual against half-installed dependencies.
$(VENV): utils/requirements.txt
	@rm -rf $(VENV)
	@{ $(PYTHON) -m venv $(VENV) && \
	   . $(VENV)/bin/activate && \
	   pip install --upgrade pip wheel > /dev/null && \
	   pip install -r $(BUILDDIR)/utils/requirements.txt; \
	 } || { rm -rf $(VENV); exit 1; }
	@touch $(VENV)

$(SPHINXCONFIG)/conf.py: $(SPHINXCONFIG)/conf.py.in
	sed -e 's,@SPARTA_SOURCE_DIR@,$(BUILDDIR)/../src,g' \
	    -e 's,@SPARTA_DOC_DIR@,$(BUILDDIR),g' $< > $@

# MathJax is fetched rather than committed, so the repository does not
# carry six megabytes of minified JavaScript and fonts.  Version 3 is used
# because it bundles its fonts in the one package; version 4 keeps them in
# a separate 13 MB package, and without it every equation renders with
# broken glyphs.  Pinned and checksummed so the build is reproducible.
$(MATHJAX):
	@echo "Fetching MathJax $(MATHJAXVER)"
	@mkdir -p $(SPHINXCONFIG)/_static
	@tmp=$$(mktemp -d); \
	 trap 'rm -rf $$tmp' EXIT; \
	 curl -sSL $(MATHJAXURL) -o $$tmp/mathjax.tgz && \
	 echo "$(MATHJAXSHA)  $$tmp/mathjax.tgz" | sha256sum -c - > /dev/null && \
	 tar -xzf $$tmp/mathjax.tgz -C $$tmp && \
	 mv $$tmp/package $(MATHJAX)

# && rather than ; here too: with ; the recipe's exit status is the status
# of the last command in the list, so a failed sphinx-build reported success
# and a broken manual looked like a good one.
html: $(SPHINXCONFIG)/conf.py $(VENV) $(MATHJAX)
	@. $(VENV)/bin/activate && \
	 sphinx-build $(SPHINXEXTRA) -b html \
	   -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) html
	@echo "Build finished. The HTML pages are in doc/html."

# The PDF is the copy that ships in the release tarball.  Sphinx writes the
# LaTeX and its own Makefile into doc/latex; that Makefile drives latexmk,
# which reruns LaTeX until the cross-references and the table of contents
# settle.
pdf: $(SPHINXCONFIG)/conf.py $(VENV)
	@if [ "$(HAS_PDFLATEX)" = "NO" ]; then \
	   echo "make pdf needs latexmk and pdflatex" 1>&2; exit 1; fi
	@. $(VENV)/bin/activate && \
	 sphinx-build $(SPHINXEXTRA) -b latex \
	   -c $(SPHINXCONFIG) -d $(BUILDDIR)/latexdoctrees $(RSTDIR) latex
	@$(MAKE) -C $(BUILDDIR)/latex LATEXMKOPTS="-interaction=nonstopmode -halt-on-error"
	@mv $(BUILDDIR)/latex/SPARTA.pdf $(BUILDDIR)/Manual.pdf
	@echo "Build finished. The manual is in doc/Manual.pdf."

# A quick look at one page while editing it: pandoc renders each .rst on its
# own, with no cross-references and no theme, in about a second.  It is a
# preview, not the manual -- "make html" is the manual.
#
# It warns on a handful of equations that use \rm or \textnormal, which its
# MathML writer does not implement, and falls back to showing them as TeX.
# Sphinx and the PDF both typeset them correctly; there is nothing to fix in
# the sources.
fasthtml: fasthtml-init $(FASTHTMLFILES)
	@echo "Preview built. The pages are in doc/fasthtml."

fasthtml-init:
	@if [ "$(HAS_PANDOC)" = "NO" ]; then \
	   echo "make fasthtml needs pandoc" 1>&2; exit 1; fi
	@mkdir -p fasthtml
	@cp -r $(RSTDIR)/JPG fasthtml/

# pandoc's MathML writer takes only em as a length unit, so a \hspace given
# in mm makes it give up on that equation.  Sphinx and MathJax both take it
# as written; substituting the thin space it stands for, only for the
# preview, keeps the equation readable without touching the source.
fasthtml/%.html: $(RSTDIR)/%.rst
	@mkdir -p fasthtml
	@sed -e 's/\\hspace{[0-9.]*mm}/\\,/g' $< > fasthtml/$*.tmp.rst
	@pandoc -s --mathml --metadata title="$*" -o $@ fasthtml/$*.tmp.rst
	@rm -f fasthtml/$*.tmp.rst
