##
# 
##

SHELL:=/bin/bash

all: install embedding

clean:
	rm -f embedding/*.zip
	rm -f embedding/*.gz
	
install: 
	@echo installing requirements 
	pip install cython
	pip install -r requirements.txt
	
update:
	@echo updating requirements
	pip install -U -r requirements.txt

forcedupdate:
	@echo updating requirements
	pip install --force-reinstall -U --upgrade-strategy eager -r requirements.txt

package:
	@echo packaging lttc
	@tar -czvf lttc.tar.gz \
	  Makefile \
	  README.md \
	  LICENSE \
	  requirements.txt \
	  lttc.py \
	  data.py \
	  modules.py \
	  embedding.py \
	  utils.py \
	  data/SMSSpamCollection/*/*/*.txt

packagemodels:
	@echo packaging lttc default models
	@tar -czvf lttc-models.tar.gz \
	  savedmodels/SMSSpamCollection_default

reinstall: clean install

embedding: embedding/cc.de.300.bin embedding/wiki.simple.bin

embedding/wiki.simple.bin:
	@read -n 1 -e -p "Download simple wikipedia fasttext embedding? [y/n]: " answer && [ ! "$${answer}" = "y" ] && echo "skipping..." || ( \
	mkdir -p embedding && cd embedding && \
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip -O wiki.simple.zip && \
	unzip wiki.simple.zip wiki.simple.bin && \
	rm wiki.simple.zip && \
	cd .. ; )

embedding/cc.de.300.bin:
	@read -n 1 -e -p "Download German CC fasttext embedding? [y/n]: " answer && [ ! "$${answer}" = "y" ] && echo "skipping..." || ( \
	mkdir -p embedding && cd embedding && \
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz -O cc.de.300.bin.gz && \
	gunzip cc.de.300.bin.gz && \
	cd .. ; )
