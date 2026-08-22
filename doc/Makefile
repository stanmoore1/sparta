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

.PHONY: help clean clean-all html

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  html          build the HTML manual in doc/html"
	@echo "  clean         remove the build tree, keep the virtualenv"
	@echo "  clean-all     also remove the virtualenv and generated conf.py"
	@echo ""
	@echo "  SPHINXEXTRA=... passes extra flags to sphinx-build, e.g."
	@echo "  make html SPHINXEXTRA=\"-j auto -W --keep-going\""

clean:
	rm -rf $(BUILDDIR)/html $(BUILDDIR)/doctrees

clean-all: clean
	rm -rf $(VENV) $(SPHINXCONFIG)/conf.py $(MATHJAX)

# the virtualenv is a directory target: it is built once, then only rebuilt
# if requirements.txt changes
$(VENV): utils/requirements.txt
	@rm -rf $(VENV)
	@( $(PYTHON) -m venv $(VENV); \
	   . $(VENV)/bin/activate; \
	   pip install --upgrade pip wheel > /dev/null; \
	   pip install -r $(BUILDDIR)/utils/requirements.txt; \
	   deactivate; )
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

html: $(SPHINXCONFIG)/conf.py $(VENV) $(MATHJAX)
	@( . $(VENV)/bin/activate; \
	   sphinx-build $(SPHINXEXTRA) -b html \
	     -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) html; \
	   deactivate; )
	@echo "Build finished. The HTML pages are in doc/html."
