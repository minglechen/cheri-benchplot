/*
 * Simple qemu trace generation test.
 * This should be cross-compiled for CHERI-QEMU architectures.
 */
#include <stdio.h>
#include <unistd.h>
#include <sys/thr.h>
#include <machine/cheri.h>
#include <machine/cpufunc.h>

/*
 * Test QEMU trace NOPs to track context identifier
 */
int
main(int argc, char *argv[])
{
  pid_t pid, tmp;
  long tid;
  int i;

  pid = getpid();
  tid = thr_self(&tid);
  printf("Test tracing pid=%d tid=%lu\n", pid, tid);

  CHERI_START_TRACE;
  QEMU_EVENT_CONTEXT_UPDATE(pid, tid, -1UL);

  /* Some fake work */
  for (i = 0; i < 50; i++)
    tmp = getpid();

  CHERI_STOP_TRACE;

  return 0;
}
