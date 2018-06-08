default:
	g++ -fopenmp -g -O2 -o bfs main.cpp bfs.cpp graph.cpp -L ./lib -lbfs

run: ./bfs
	LD_LIBRARY_PATH=./lib/:$(LD_LIBRARY_PATH) ./bfs ${GRAPH} $(N_THREADS)

clean:
	rm -rf bfs *.o *~ *.*~
