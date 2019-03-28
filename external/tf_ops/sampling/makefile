nvcc = /usr/local/cuda/bin/nvcc
cudalib = /usr/local/cuda/lib64
tensorflow = /usr/local/lib/python3.5/dist-packages/tensorflow/include
TF_LIB = /usr/local/lib/python3.5/dist-packages/tensorflow

all: tf_sampling_g.cu.o tf_sampling_so.so 

tf_sampling_g.cu.o:
	$(nvcc) tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_sampling_so.so: tf_sampling_g.cu.o
	g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $(tensorflow) -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L $(TF_LIB) -ltensorflow_framework

clean:
	rm *.so
	rm  *.o 
