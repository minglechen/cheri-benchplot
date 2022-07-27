#!/bin/sh

${CHERIBUILD} --skip-update cheribsd-riscv64-purecap
${CHERIBUILD} --skip-update disk-image-mfsroot-riscv64-purecap
${CHERIBUILD} --skip-update --skip-buildworld cheribsd-mfs-root-kernel-riscv64-purecap --cheribsd/build-bench-kernels --cheribsd/build-fpga-kernels --cheribsd/extra-kernel-config "CHERI-GFE-BUCKET-ADJUST CHERI-PURECAP-GFE-BUCKET-ADJUST CHERI-PURECAP-NOSUBOBJ-GFE-NODEBUG"
