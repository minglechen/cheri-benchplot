
#include <stdio.h>

struct bar {
  int x;
  int y;
};

typedef struct baz {
  long z;
} baz_t;

struct forward;

struct foo {
  int a;
  char *b;
  struct bar c;
  const char *d;
  char * const e;
  const volatile void *f;
  int **g;
  int *h[10];
  void (*i)(int, int);
  int (*j)[10];
  baz_t k;
  struct forward *l;
};

void show(struct foo *fp)
{
  printf("Foo %d %s\n", fp->a, fp->b);
}

int main(int argc, char *argv[])
{

  struct foo f;

  f.a = 10;
  f.b = "hello";
  show(&f);

  return 0;
}
