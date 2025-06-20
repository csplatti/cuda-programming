# Parallel Reduction
## What is it?
Say you have a list of integers
```c++
    int nums[] = {1, 2, 3, 4, 5, 6, 7};
```

For many problems, it can be useful to **reduce** such a list of numbers down to a single value (their sum, product, max, min, etc.). Parallel reduction aims to paralellize this class of problem for better performance.

## My Implementation