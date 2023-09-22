
struct bar {
};

struct foo {
  int a;
  struct bar b;
};

int main(int argc, char *argv[])
{
  struct foo f;

  return 0;
}
