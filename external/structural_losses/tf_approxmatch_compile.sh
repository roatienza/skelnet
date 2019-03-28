set -e
if [ 'tf_approxmatch_g.cu.o' -ot 'tf_approxmatch_g.cu' ] ; then
	echo 'nvcc'
	/usr/local/cuda/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
fi
if [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch.cpp' ] || [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch_g.cu.o' ] ; then
	echo 'g++'
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /usr/local/cuda/include  -L /usr/local/cuda/lib64/ -O2 -I /usr/local/lib/python3.5/dist-packages/tensorflow/include 
fi

