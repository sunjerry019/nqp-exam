MKLROOT="/software/opt/focal/x86_64/spack/2021.12/spack/opt/spack/linux-ubuntu20.04-x86_64_v2/gcc-9.3.0/intel-oneapi-mkl-2021.4.0-finhpnl2dqry3iuhcvtrpwagvgvngpl5/mkl/2021.4.0"
LIBMKL="$MKLROOT/lib"
INTELMKL="$LIBMKL/intel64"
source $MKLROOT/env/vars.sh -arch intel64

module purge
if ! $(module list | grep -q spack/2021.12); then module load spack/2021.12; fi
if ! $(module list | grep -q intel-oneapi-mkl/2021.4.0); then module load intel-oneapi-mkl/2021.4.0; fi
if ! $(module list | grep -q python/3.9-2021.11); then module load python/3.9-2021.11; fi
