
CHERISDK?=~/cherisdk

CHERI_CC?=$(CHERISDK)/sdk/bin/clang
CHERI_CFLAGS=--sysroot $(CHERISDK)/rootfs-riscv64-purecap -target riscv64-unknown-freebsd -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax

SOURCES := $(wildcard test_*.c)
CROSS_SOURCES := $(wildcard qemu_*.c)

TARGETS := $(SOURCES:%.c=%)
CROSS_TARGETS := $(CROSS_SOURCES:%.c=riscv_%)
# $(CROSS_SOURCES:%.c:morello_%)

all: $(TARGETS) $(CROSS_TARGETS)

%: %.c
	gcc -gdwarf-4 -o $@ $<

riscv_%: %.c %.S
	$(CHERI_CC) $(CHERI_CFLAGS) -o $@ $^

benchplot-assets:
	python gen_benchplot_assets.py
