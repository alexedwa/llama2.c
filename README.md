# COMP3000 Final Project
## High-Performance Optimisation of an LLM

This is the optimised version of the open source [llama2.c](https://github.com/karpathy/llama2.c) LLM.

## Run

To run the optimised version of the LLM you will need to:
- Clone the repository into a project folder
  - `git clone https://github.com/alexedwa/llama2.c.git`

- Then compile the file with either the *Makefile*
  - `make run`

- Or compile via the terminal
  - `gcc run.c win.c -o llama2.exe -mavx2 -mfma -lm -O3 -fopenmp -march=native`

- Then finally run the file
  - `./llama2.exe`

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