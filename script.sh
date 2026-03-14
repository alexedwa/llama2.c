export GOMP_CPU_AFFINITY="0,2,4,6"
export OMP_NUM_THREADS=4
echo "Physical performance cores only"
./matmul.exe

export GOMP_CPU_AFFINITY="0-7"
export OMP_NUM_THREADS=8
echo "Physical performance with hyper threading cores"
./matmul.exe

export GOMP_CPU_AFFINITY="8,9,10,11" 
export OMP_NUM_THREADS=4
echo "Efficient cores only"
./matmul.exe


export GOMP_CPU_AFFINITY="0-11" 
export OMP_NUM_THREADS=4
echo "All cores"
./matmul.exe