
CHERISDK?=~/cherisdk

CHERI_CC?=$(CHERISDK)/sdk/bin/clang
CHERI_CFLAGS=--sysroot $(CHERISDK)/rootfs-riscv64-purecap -target riscv64-unknown-freebsd -march=rv64imafdcxcheri -mno-relax

SOURCES :=
CROSS_SOURCES := $(wildcard test_*.c)$(wildcard qemu_*.c)

TARGETS := $(SOURCES:%.c=%)
CROSS_TARGETS := $(CROSS_SOURCES:%.c=riscv_hybrid_%) $(CROSS_SOURCES:%.c=riscv_purecap_%)

all: $(TARGETS) $(CROSS_TARGETS)

%: %.c
	gcc -gdwarf-4 -o $@ $<

riscv_hybrid_%: %.c
	$(CHERI_CC) $(CHERI_CFLAGS) -gdwarf-4 -mabi=lp64d -o $@ $^

riscv_purecap_%: %.c
	$(CHERI_CC) $(CHERI_CFLAGS) -gdwarf-4 -mabi=l64pc128d -o $@ $^

riscv_%: %.c %.S
	$(CHERI_CC) $(CHERI_CFLAGS) -mabi=l64pc128d -o $@ $^

benchplot-assets:
	python gen_benchplot_assets.py

clean:
	rm $(TARGETS) $(CROSS_TARGETS)
