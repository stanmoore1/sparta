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
	rm -rf $(VENV) $(SPHINXCONFIG)/conf.py

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

html: $(SPHINXCONFIG)/conf.py $(VENV)
	@( . $(VENV)/bin/activate; \
	   sphinx-build $(SPHINXEXTRA) -b html \
	     -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) html; \
	   deactivate; )
	@echo "Build finished. The HTML pages are in doc/html."
