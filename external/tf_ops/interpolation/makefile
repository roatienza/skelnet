nvcc = /usr/local/cuda/bin/nvcc
cudalib = /usr/local/cuda/lib64/
cudainc = /usr/local/cuda/include
tensorflow = /usr/local/lib/python3.5/dist-packages/tensorflow/include
TF_LIB = /usr/local/lib/python3.5/dist-packages/tensorflow

all: tf_interpolate_so.so
.PHONY : all

tf_interpolate_so.so: tf_interpolate.cpp
	g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $(tensorflow)  -I $(cudainc) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
