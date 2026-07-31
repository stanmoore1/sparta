# Makefile for SPARTA documentation
#
# Adapted from the LAMMPS doc/Makefile. The doxygen, graphviz, globbed-toc
# and package-check machinery is omitted: SPARTA has no programmer guide
# generated from source comments, no style-globbed toctrees, and no packages.

SHELL          = /bin/bash
HAS_BASH       = YES
ifeq (,$(wildcard $(SHELL)))
OSHELL         := $(SHELL)
override SHELL = /bin/sh
HAS_BASH       = NO
endif
BUILDDIR       = ${CURDIR}
RSTDIR         = $(BUILDDIR)/src
VENV           = $(BUILDDIR)/docenv
ANCHORCHECK    = $(VENV)/bin/rst_anchor_check
SPHINXCONFIG   = $(BUILDDIR)/utils/sphinx-config
MATHJAX        = $(SPHINXCONFIG)/_mathjax/mathjax
MATHJAXTAG     = 4.1.3

PYTHON         = $(word 3,$(shell type python3))
PANDOC         = $(word 3,$(shell type pandoc))
HAS_PYTHON3    = NO
HAS_PDFLATEX   = NO
HAS_PANDOC     = NO

ifeq ($(shell type python3 >/dev/null 2>&1; echo $$?), 0)
HAS_PYTHON3    = YES
endif

ifeq ($(shell type pdflatex >/dev/null 2>&1; echo $$?), 0)
ifeq ($(shell type latexmk >/dev/null 2>&1; echo $$?), 0)
HAS_PDFLATEX   = YES
endif
endif

ifeq ($(shell type pandoc >/dev/null 2>&1; echo $$?), 0)
HAS_PANDOC     = YES
endif

# override settings for PIP commands, e.g. behind a proxy or a corporate CA
# PIP_OPTIONS = --cert /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt --proxy http://proxy.mydomain.org

SPHINXEXTRA = -j $(shell $(PYTHON) -c 'import multiprocessing;print(multiprocessing.cpu_count())')

.PHONY: help clean-all clean clean-spelling clean-linkcheck epub html pdf \
        spelling anchor_check style_check char_check link_check role_check \
        fasthtml fasthtml-init check upgrade

FASTHTMLFILES = $(patsubst $(RSTDIR)/%.rst,fasthtml/%.html,$(wildcard $(RSTDIR)/*rst))
# ------------------------------------------

help:
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html          create HTML pages in html dir"
	@echo "  pdf           create Manual.pdf in this dir"
	@echo "  epub          create ePUB format manual for e-book readers"
	@echo "  fasthtml      approximate HTML page creation in fasthtml dir (for development)"
	@echo "  upgrade       upgrade sphinx, extensions, and dependencies to latest supported versions"
	@echo "  clean         remove all intermediate files"
	@echo "  clean-all     reset the entire build environment"
	@echo "  check         run 'anchor_check', 'style_check', 'char_check', and 'role_check'"
	@echo "  anchor_check  scan for duplicate anchor labels"
	@echo "  style_check   check for complete and consistent style lists"
	@echo "  char_check    check for non-ASCII characters"
	@echo "  role_check    check for misformatted role keywords"
	@echo "  link_check    check external links in the manual"
	@echo "  spelling      spell-check the manual"

# ------------------------------------------

clean-all: clean
	rm -rf $(BUILDDIR)/docenv $(MATHJAX) $(BUILDDIR)/SPARTA.epub $(BUILDDIR)/Manual.pdf

clean: clean-spelling clean-linkcheck
	rm -rf $(BUILDDIR)/html $(BUILDDIR)/epub $(BUILDDIR)/latex $(BUILDDIR)/doctrees \
	       $(BUILDDIR)/doctrees-spelling $(SPHINXCONFIG)/conf.py $(BUILDDIR)/fasthtml \
	       $(BUILDDIR)/spelling

clean-spelling:
	rm -rf $(BUILDDIR)/spelling

clean-linkcheck:
	rm -rf $(BUILDDIR)/linkcheck $(BUILDDIR)/linkchecker-out.html

$(SPHINXCONFIG)/conf.py: $(SPHINXCONFIG)/conf.py.in
	sed -e 's,@SPARTA_SOURCE_DIR@,$(BUILDDIR)/../src,g' \
	    -e 's,@SPARTA_PYTHON_DIR@,$(BUILDDIR)/../python,g' \
	    -e 's,@SPARTA_DOC_DIR@,$(BUILDDIR),g' $< > $@

html: $(VENV) $(SPHINXCONFIG)/conf.py $(ANCHORCHECK) $(MATHJAX)
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@(\
		. $(VENV)/bin/activate ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build -E $(SPHINXEXTRA) -b html -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) html ;\
		ln -sf Manual.html html/index.html;\
		echo "############################################" ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		rst_anchor_check src/*.rst ;\
		env LC_ALL=C grep -n '[^ -~]' $(RSTDIR)/*.rst ;\
		env LC_ALL=C grep -n ' :[a-z]\+`' $(RSTDIR)/*.rst ;\
		env LC_ALL=C grep -n ' `[^`]\+<[a-z][^`]\+`[^_]' $(RSTDIR)/*.rst ;\
		env LC_ALL=C grep -n ':\(ref\|doc\):[^`]' $(RSTDIR)/*.rst ;\
		env LC_ALL=C grep -n '\(ref\|doc\)`[^`]' $(RSTDIR)/*.rst ;\
		$(PYTHON) $(BUILDDIR)/utils/check-styles.py -s ../src -d src ;\
		env LC_ALL=C grep -n -E '^ *\.\. [a-z0-9]+:(\s+.*|)$$' \
		                 $(RSTDIR)/*.rst ../src/*.{cpp,h} ../src/*/*.{cpp,h} ;\
		echo "############################################" ;\
		deactivate ;\
	)
	@rm -rf html/_sources
	@echo "Build finished. The HTML pages are in doc/html."

fasthtml: fasthtml-init $(FASTHTMLFILES)
	@echo "Fast HTML build finished. The HTML pages are in doc/fasthtml."

fasthtml-init:
	@mkdir -p fasthtml/JPG
	@cp src/JPG/*.* fasthtml/JPG
	@cp $(BUILDDIR)/utils/pandoc.css fasthtml/

fasthtml/%.html: $(RSTDIR)/%.rst
	@if [ "$(HAS_PANDOC)" == "NO" ] ; then echo "Make 'fasthtml' requires the 'pandoc' software" 1>&2; exit 1; fi
	@mkdir -p fasthtml
	@echo converting $< to $@
	@pandoc -s --mathml --css="pandoc.css" --template=$(BUILDDIR)/utils/pandoc.html --metadata title="$@" -o $@ $<

# use a dedicated doctree cache: the html/epub/pdf builds use -E (which would
# discard it) and produce different doctrees (e.g. from the sphinx-tabs hack
# for non-html builders), so sharing $(BUILDDIR)/doctrees would re-read all
# sources on every run instead of only changed files.
spelling: $(SPHINXCONFIG)/conf.py $(VENV) $(SPHINXCONFIG)/false_positives.txt
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@rm -rf $(BUILDDIR)/spelling
	@(\
		. $(VENV)/bin/activate ; \
		cp $(SPHINXCONFIG)/false_positives.txt $(RSTDIR)/;  env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build $(SPHINXEXTRA) -b spelling -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees-spelling $(RSTDIR) spelling ;\
		deactivate ;\
	)
	@echo "Spell check finished."

link_check: $(SPHINXCONFIG)/conf.py $(VENV) $(SPHINXCONFIG)/false_positives.txt
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@(\
		. $(VENV)/bin/activate ; \
		cp $(SPHINXCONFIG)/false_positives.txt $(RSTDIR)/;  env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build -b linkcheck -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) linkcheck ;\
		deactivate ;\
	)
	@echo "Link check finished."

epub: $(VENV) $(SPHINXCONFIG)/conf.py $(ANCHORCHECK)
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@mkdir -p epub/JPG
	@rm -f SPARTA.epub
	@cp src/JPG/*.* epub/JPG
	@(\
		. $(VENV)/bin/activate ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build -E $(SPHINXEXTRA) -b epub -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) epub ;\
		deactivate ;\
	)
	@mv epub/SPARTA.epub .
	@rm -rf epub
	@echo "Build finished. The ePUB manual file is created."

pdf: $(VENV) $(SPHINXCONFIG)/conf.py $(ANCHORCHECK)
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@if [ "$(HAS_PDFLATEX)" == "NO" ] ; then echo "PDFLaTeX or latexmk were not found! Please check README for further instructions" 1>&2; exit 1; fi
	@(\
		. $(VENV)/bin/activate ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build -E $(SPHINXEXTRA) -b latex -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) latex ;\
		echo "############################################" ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		rst_anchor_check src/*.rst ;\
		env LC_ALL=C grep -n '[^ -~]' $(RSTDIR)/*.rst ;\
		$(PYTHON) $(BUILDDIR)/utils/check-styles.py -s ../src -d src ;\
		echo "############################################" ;\
		deactivate ;\
	)
	@cd latex && \
		sed 's/\\begin{equation}//g' SPARTA.tex > tmp.tex && \
		mv tmp.tex SPARTA.tex && \
		sed 's/\\end{equation}//g' SPARTA.tex > tmp.tex && \
		mv tmp.tex SPARTA.tex && \
		sed 's/\\contentsname}{.*}}/\\contentsname}{SPARTA Documentation}}/g' SPARTA.tex > tmp.tex && \
		mv tmp.tex SPARTA.tex && \
		$(MAKE) $(MFLAGS) && \
		mv SPARTA.pdf ../Manual.pdf && \
		cd ../;
	@rm -rf latex/_sources
	@echo "Build finished. Manual.pdf is in this directory."

check : anchor_check style_check char_check role_check

anchor_check : $(ANCHORCHECK)
	@(\
		. $(VENV)/bin/activate ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		rst_anchor_check src/*.rst ;\
		deactivate ;\
	)

style_check : $(VENV)
	@(\
		. $(VENV)/bin/activate ; env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) utils/check-styles.py -s ../src -d src ;\
		deactivate ;\
	)

char_check :
	@( env LC_ALL=C grep -n '[^ -~]' $(RSTDIR)/*.rst && exit 1 || : )

role_check :
	@( env LC_ALL=C grep -n ' :[a-z]\+`' $(RSTDIR)/*.rst && exit 1 || : )
	@( env LC_ALL=C grep -n ' `[^`]\+<[a-z][^`]\+`[^_]' $(RSTDIR)/*.rst && exit 1 || : )
	@( env LC_ALL=C grep -n ':\(ref\|doc\):[^`]' $(RSTDIR)/*.rst && exit 1 || : )
	@( env LC_ALL=C grep -n '\(ref\|doc\)`[^`]' $(RSTDIR)/*.rst && exit 1 || : )
	@( env LC_ALL=C grep -n -E '^ *\.\. [a-z0-9]+:(\s+.*|)$$' \
	                $(RSTDIR)/*.rst ../src/*.{cpp,h} ../src/*/*.{cpp,h} && exit 1 || : )

upgrade: $(VENV)
	@(\
		. $(VENV)/bin/activate; \
		pip $(PIP_OPTIONS) install --upgrade pip; \
		pip $(PIP_OPTIONS) install --upgrade wheel; \
		pip $(PIP_OPTIONS) install --upgrade -r $(BUILDDIR)/utils/requirements.txt; \
		deactivate;\
	)

# ------------------------------------------

$(VENV):
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@if [ "$(HAS_PYTHON3)" == "NO" ] ; then echo "python3 was not found! Please see README for further instructions" 1>&2; exit 1; fi
	@( \
		$(PYTHON) -m venv $(VENV); \
		. $(VENV)/bin/activate; \
		pip $(PIP_OPTIONS) install --upgrade pip; \
		pip $(PIP_OPTIONS) install --upgrade wheel; \
		pip $(PIP_OPTIONS) install -r $(BUILDDIR)/utils/requirements.txt; \
		deactivate;\
	)

$(ANCHORCHECK): $(VENV)
	@( . $(VENV)/bin/activate; \
	   pip $(PIP_OPTIONS) install -e utils/converters; deactivate; )

# MathJax is cloned locally rather than loaded from a CDN so the built manual
# renders equations with no network access.
$(MATHJAX):
	@git clone -b $(MATHJAXTAG) -c advice.detachedHead=0 --depth 1 https://github.com/mathjax/MathJax.git $@
