# Example Kernels

## Matmul

```
def matmul(C: (M, N), A: (M, K), B: (K, N)):
  loop "i" (0, M) 1 in
    loop "j" (0, N) 1 in
      loop "k" (0, K) 1 in
        C[i, j] = C[i, j] + A[i, k] * B[k, j]
```

## Transpose

```
def transpose(A.T: (N, M), A: (M, N)):
  loop "i" (0, N) 1 in
    loop "j" (0, M) 1 in
      A.T[i, j] = A[j, i]
```

## Element-Wise Addition

```
def ewise_add(C: (M, N), A: (M, N), B: (M, N)):
  loop "i" (0, M) 1 in
    loop "j" (0, N) 1 in
      C[i, j] = A[i, j] + B[i, j]
```

## Reduce Sum

```
def reduce_sum(A: (M, N)):
  let sum = 0 in
  loop "i" (0, M) 1 in
    loop "j" (0, N) 1 in
      sum += A[i, j]
  in sum
```

## Reduce Sum (Axis)

```
def reduce_sum(A.O: (M,), A: (M, N)):
  loop "i" (0, M) 1 in
    loop "j" (0, N) 1 in
      A.O[i] = A.O[i] + A[i, j]
```

## MaxPool (with 0 padding)

```
def maxpool2d(A.O: (N, C', H', W'), A.I: ())
```

## Conv2D (with 0 padding)

```
def conv2d(A.O: (N, C', H', W'), K: (C', C, K, K), A.I: (N, C, H, W)):
  define "K.MID" = K / 2 in
  loop "n" (0, N) 1 in
    loop "c'" (0, C') 1 in
      loop "i'" (0, H') 1 in
        loop "j'" (0, W') 1 in
          loop "c" (0, C) 1 in
            loop "k.i" (0, K) 1 in
              loop "k.j" (0, K) 1 in
                A.O[n, c', i', j'] = A.O[n, c', i', j'] +
                  K[c', c, i, j] * A.I[n, c, i' + k.i - K.MID, j' + k.j - K.MID]
```
