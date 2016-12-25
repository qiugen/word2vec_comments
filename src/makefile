CC = gcc
PP = g++
#Using -O2 instead of -O3 might result in faster code, but is supported only by newer GCC versions
#CFLAGS = -lm -pthread -O3 -Wall -funroll-loops 
CFLAGS = -lm -pthread -g -Wall -funroll-loops 
OUTPATH = ../bin/
#all: word2vec word2phrase  distance distance_out word-analogy compute-accuracy bin2txt
all: word2vec bin2txt distance_search distance txt2bin get_best_candi dump_corpus

word2vec : word2vec.c
	$(CC) word2vec.c -o $(OUTPATH)word2vec $(CFLAGS)
#word2phrase : word2phrase.c
#	$(CC) word2phrase.c -o $(OUTPATH)word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o $(OUTPATH)distance $(CFLAGS)
dump_corpus : dump_corpus.c
	$(PP) dump_corpus.c -o $(OUTPATH)dump_corpus $(CFLAGS)
distance_search : distance_search.c
	$(CC) distance_search.c -o $(OUTPATH)distance_search $(CFLAGS)
#distance_out : distance_out.cpp
#	$(PP) distance_out.cpp -o $(OUTPATH)distance_out $(CFLAGS)
#word-analogy : word-analogy.c
#	$(CC) word-analogy.c -o $(OUTPATH)word-analogy $(CFLAGS)
get_best_candi: get_best_candi.c
	$(PP) get_best_candi.c -o $(OUTPATH)get_best_candi $(CFLAGS)
txt2bin : txt2bin.c
	$(PP) txt2bin.c -o $(OUTPATH)txt2bin $(CFLAGS)
bin2txt : bin2txt.c
	$(PP) bin2txt.c -o $(OUTPATH)bin2txt $(CFLAGS)
#compute-accuracy : compute-accuracy.c
#	$(CC) compute-accuracy.c -o $(OUTPATH)compute-accuracy $(CFLAGS)
#	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy bin2txt distance_out
