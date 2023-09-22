
union nested_union {
  int x;
  long y;
};

struct bar {
  int x;
  union nested_union u;
};

struct baz {
  int v;
  union nested_union u;
};

struct foo {
  int a;
  union {
    struct bar b_bar;
    struct baz b_baz;
  } b;
};

int main(int argc, char *argv[])
{
  struct foo f;

  return 0;
}
