nvcc = /usr/local/cuda/bin/nvcc
cudalib = /usr/local/cuda/lib64/
tensorflow = /usr/local/lib/python3.5/dist-packages/tensorflow/include
TF_LIB = /usr/local/lib/python3.5/dist-packages/tensorflow

all: tf_auctionmatch_so.so
.PHONY : all

tf_auctionmatch_so.so: tf_auctionmatch.cpp tf_auctionmatch_g.cu.o
	g++ -std=c++11 tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $(tensorflow)  -I /usr/local/cuda/include -lcudart -L $(cudalib) -O2 -L $(TF_LIB) -ltensorflow_framework

tf_auctionmatch_g.cu.o: tf_auctionmatch_g.cu
	$(nvcc) tf_auctionmatch_g.cu -o tf_auctionmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_30 

clean:
	rm *.so
	rm  *.o 
