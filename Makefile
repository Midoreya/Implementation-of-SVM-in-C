BIN=test

CC=gcc
CXX=g++

CFLAGS=-Wall -Wextra

# LIBS= -ljpeg -lm
LIBS= -lm -lopenblas
SRCS=$(wildcard *.c)
OBJS=$(addsuffix .o, $(basename $(SRCS)))

.SUFFIXES: .c
.c.o:
	$(CC) $(CFLAGS) -c $<

all: $(BIN) 

$(BIN): $(OBJS)
	$(CXX) $(CFLAGS) $(LFLAGS) $(EXTRALFLAGS) $^  $(LIBS) -o $@

.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(BIN)
	rm -rf *~
