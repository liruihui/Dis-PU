#/bin/bash

TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_ROOT=/usr/local/cuda-10.2


g++ -std=c++11 -shared ./interpolation/tf_interpolate.cpp -o ./interpolation/tf_interpolate_so.so  -I $CUDA_ROOT/include -lcudart -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

$CUDA_ROOT/bin/nvcc ./grouping/tf_grouping_g.cu -o ./grouping/tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./grouping/tf_grouping.cpp ./grouping/tf_grouping_g.cu.o -o ./grouping/tf_grouping_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

$CUDA_ROOT/bin/nvcc ./sampling/tf_sampling_g.cu -o ./sampling/tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./sampling/tf_sampling.cpp ./sampling/tf_sampling_g.cu.o -o ./sampling/tf_sampling_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

$CUDA_ROOT/bin/nvcc ./approxmatch/tf_approxmatch_g.cu -o ./approxmatch/tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./approxmatch/tf_approxmatch.cpp ./approxmatch/tf_approxmatch_g.cu.o -o ./approxmatch/tf_approxmatch_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

$CUDA_ROOT/bin/nvcc ./nn_distance/tf_nndistance_g.cu -o ./nn_distance/tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./nn_distance/tf_nndistance.cpp ./nn_distance/tf_nndistance_g.cu.o -o ./nn_distance/tf_nndistance_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

cd ../
