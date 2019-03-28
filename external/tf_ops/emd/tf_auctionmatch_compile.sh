echo 'nvcc'
/usr/local/cuda/bin/nvcc tf_auctionmatch_g.cu -o tf_auctionmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_30 
echo 'g++'
g++ -std=c++11 tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I /usr/local/lib/python3.5/dist-packages/tensorflow/include  -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -L /usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework

