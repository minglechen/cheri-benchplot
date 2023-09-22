/*
 * Simple qemu trace generation test.
 * This should be cross-compiled for CHERI-QEMU architectures.
 */
#include <unistd.h>
#include <machine/cheri.h>
#include <machine/cpufunc.h>

/*
 * Test QEMU trace NOPs to start and stop tracing.
 */
int
main(int argc, char *argv[])
{
  pid_t tmp;
  int i;

  CHERI_START_TRACE;

  /* Some fake work */
  for (i = 0; i < 50; i++)
    tmp = getpid();

  CHERI_STOP_TRACE;

  return 0;
}
