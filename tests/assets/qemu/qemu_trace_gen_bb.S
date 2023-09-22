
.text
.global test_bb
test_bb:
    li  a1, 50
    li  a0, 0
1:
    addi    a0, a0, 1
    addi    a1, a1, -1
    bnez    a1, 1b
    cret
.size test_bb, . - test_bb
