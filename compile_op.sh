cd libs/nearest_neighbors
python setup.py install --home="."
cd ../../

cd libs/cpp_wrappers
sh compile_wrappers.sh
cd ../../

cd tf_ops/
sh compile_ops.sh
