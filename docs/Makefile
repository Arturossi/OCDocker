# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
.PHONY: help
help:
	$(SPHINXBUILD) --help

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

.PHONY: latexpdf
latexpdf:
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/latex
	$(MAKE) -C $(BUILDDIR)/latex all-pdf

.PHONY: epub
epub:
	$(SPHINXBUILD) -b epub $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/epub

.PHONY: linkcheck
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck

.PHONY: doctest
doctest:
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/doctest

.PHONY: livehtml
livehtml:
	$(SPHINXAUTOBUILD) $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

.PHONY: all
all: clean html latexpdf epub

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -b $@ $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/$@
