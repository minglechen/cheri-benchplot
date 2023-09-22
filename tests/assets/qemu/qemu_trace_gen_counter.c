/*
 * Test qemu/perfetto counter events.
 * This should be cross-compiled for CHERI-QEMU architectures.
 */
#include <stdio.h>
#include <unistd.h>
#include <machine/cheri.h>
#include <machine/cpufunc.h>

/*
 * Test QEMU trace NOPs to emit trace counters.
 */
int
main(int argc, char *argv[])
{
  pid_t tmp;
  int i;

  CHERI_START_TRACE;

  /* Some fake work */
  for (i = 0; i < 50; i++) {
    QEMU_EVENT_COUNTER("test-counter", 1);
    tmp = getpid();
  }

  CHERI_STOP_TRACE;

  return 0;
}
