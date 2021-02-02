# How to compile gather.cpp

```bash
# Load gcc, cuda and openmpi
module load gcc/8.4.0 cuda/11.1.0 openmpi/4.0.5_slurm

# Compile with mpicxx, notice -lcudart
mpicxx -g gather.cpp -o gather -I/opt/psi/Programming/cuda/11.1.0/lib64 -L/opt/psi/Programming/cuda/11.1.0/lib64 -lcudart
```
