
#include <stdio.h>

/* lifted from pahole output, 8 bytes pointers */
struct foo {
  char                       why_l_[0];            /*     0     0 */
  const char  *              why;                  /*     0     8 */
  char                       why_r_[0];            /*    8     0 */
  char                       nargs_l_[0];          /*    8     0 */
  int                        nargs;                /*    8     4 */
  char                       nargs_r_[12];         /*    12    12 */
  char                       args_l_[0];           /*    24     0 */
  void * *                   args;                 /*    24     8 */
  char                       args_r_[0];            /*    32     0 */
};

void show(struct foo *fp)
{
  printf("Foo %d\n", fp->nargs);
}

int main(int argc, char *argv[])
{

  struct foo f;

  f.nargs = 10;
  show(&f);

  return 0;
}
