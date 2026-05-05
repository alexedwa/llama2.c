# COMP3000 Final Project
## High-Performance Optimisation of an LLM

This is the optimised version of the open source [llama2.c](https://github.com/karpathy/llama2.c) LLM.

## Run
### CPU Implementation
To run the optimised version of the LLM on a CPU you will need to:
- Clone the repository into a project folder
  - `git clone https://github.com/alexedwa/llama2.c.git`

- Then compile the file with either the *Makefile*
  - `make run`

- Or compile via the terminal
  - `gcc run.c win.c -o llama2.exe -mavx2 -mfma -lm -O3 -fopenmp -march=native`
 
- Download dataset from either [HuggingFace](https://huggingface.co/) or any other dataset repository

- Then finally run the file
  - `./llama2.exe <dataset.bin> <arguments>`
 
### GPU Implementation
To run the optimised version of the LLM on a GPU you will need to:
- Clone the repository into a project folder
  - `git clone https://github.com/alexedwa/llama2.c.git`

- Compile via the terminal
  - `nvcc run.cu win.c -o llama2.exe -O3`
 
- Download dataset from either [HuggingFace](https://huggingface.co/) or any other dataset repository

- Then finally run the file
  - `./llama2.exe <dataset.bin> <arguments>`

## Changelog
- 13/11/25
  - Added simple AVX2 vectorisation to the matmul function in run.c.

- 19/11/25
  - Applied register blocking to the matmul routine.

- 20/11/25
  - Applied loop tiling to the matmul routine.

- 5/12/25
  - Applied parallelism using OpenMP to matmul. 
  - Included a file just for testing the matmul function.

- 6/12/25
  - Changed SIMD loading values from being misaligned to aligned. (_mm_loadu_ps -> _mm_load_ps)
  - Changed OMP scheduling from dynamic to static.
  - Included nowait OMP clause.

- 18/01/26
  - Changed register blocking factor from 4 to 8.
  - Added array boundary checks for higher register blocking factors.

- 19/01/26
  - Added CUDA testing enviroment for the matmul function.

- 23/01/26
  - Increased register blocking factor from 8 to 16, increasing GFLOPs.

- 31/01/26
  - Created a bash script to test the efficient and performance CPU cores

- 14/03/26
  - Created different OpenMP functions to test different levels of OpenMP usages

- 3/04/26  
  - Altered run.cu to help data transfer bottlenecks
  - Added register blocking to CUDA matmul.cu implementation
 
- 17/04/26
  - Added loop tiling to CUDA Implementation

- 18/04/26
  - Added software pipelining to CUDA implementation
 
- 20/04/26
  - Applied software piplining + loop tiling implementation to main run.cu file
