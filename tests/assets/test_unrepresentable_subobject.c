
struct test_large_subobject {
  int skew_offset;
  char large_buffer[((1 << 13) - 7)];
};

struct test_small_subobject {
  int int_value;
  long long_value;
  char small_buffer[256];
  void *pointer_value;
};

struct test_age_softc_layout {
  char before[0x250];
  char cdata[0x6140];
  char after[0x190];
};

struct test_complex {
  int before;
  struct {
    char buf_before[(1 << 13) - 4];
    // 8KiB layout boundary
    char buf_after[(1 << 13) - 5];
  } inner;
  char after[10];
};

struct test_flexible {
  int int_value;
  char flexbuf[];
};

int main() {
  struct test_large_subobject a;
  struct test_small_subobject b;
  struct test_age_softc_layout c;
  struct test_flexible d;
  struct test_complex e;

  return 0;
}
