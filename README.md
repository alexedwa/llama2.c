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
  - `gcc run.c win.c -o llama2.exe -mavx2 -mfma -lm -O3`

- Then finally run the file
  - `./run.exe`

## Changelog
- 13/11/25
  - Added simple AVX2 vectorisation to the run.c file