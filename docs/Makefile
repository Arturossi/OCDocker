# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = build
IMAGE_SOURCE = OCDocker.png
IMAGE_TARGET = $(SOURCEDIR)/_static/OCDocker.png

# Put it first so that "make" without argument is like "make help".
.PHONY: help
help:
	$(SPHINXBUILD) --help

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
	rm -f $(SOURCEDIR)/_static/OCDocker.png

.PHONY: html
html: assets
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

.PHONY: latexpdf
latexpdf: assets
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/latex
	$(MAKE) -C $(BUILDDIR)/latex all-pdf

.PHONY: epub
epub: assets
	$(SPHINXBUILD) -b epub $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/epub

.PHONY: linkcheck
linkcheck: assets
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck

.PHONY: doctest
doctest: assets
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/doctest

.PHONY: livehtml
livehtml: assets
	$(SPHINXAUTOBUILD) $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

.PHONY: assets
assets:
	@mkdir -p $(dir $(IMAGE_TARGET))
	@if [ -f $(IMAGE_SOURCE) ]; then \
		cp $(IMAGE_SOURCE) $(IMAGE_TARGET); \
	else \
		echo "Note: $(IMAGE_SOURCE) not found, skipping image copy."; \
	fi

.PHONY: all
all: clean html latexpdf epub

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -b $@ $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/$@
