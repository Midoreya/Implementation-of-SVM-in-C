BIN=test

CC=gcc 
CXX=g++

CFLAGS=-Wall -Wextra -O3

LIBS= -lm -lopenblas
SRCS=$(wildcard *.c)
OBJS=$(addsuffix .o, $(basename $(SRCS)))

.SUFFIXES: .c
.c.o:
	$(CC) $(CFLAGS) -c $<

all: $(BIN) 

$(BIN): $(OBJS)
	$(CXX) $(CFLAGS) $^  $(LIBS) -o $@

.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(BIN)
	rm -rf *~
