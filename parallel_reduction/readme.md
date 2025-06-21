# Parallel Reduction
## What is it?
Say you have a list of integers
```c++
    int nums[] = {1, 2, 3, 4, 5, 6, 7};
```

For many problems, it can be useful to **reduce** such a list of numbers down to a single value (their sum, product, max, min, etc.). Parallel reduction aims to paralellize this class of problem for better performance.

## My First Implementation

The strategy for my first attempt was as follows where $N$ is the length of the input array:

1) Create $\lceil\frac{N}{2}\rceil$ threads so that the first half of the elements in the list are indexed to
2) Add the number at index $i + \lceil\frac{N}{2}\rceil$ to the number at index $i$ and store it in the array
3) Recursively call the kernel on the first half of the list with threads assigned to half of the indeces, unless the first half of the list only contains one element.

The following results are for adding up all the integers from 1 to 1024:
```bash
serial: 524800 Runtime: 0ms
Parallel: 524800 Runtime: 26ms
```

I was happy to have a solution that worked, but objectively this wasnt a good solution for a number of reasons.

### 1) Recursion
Recursion is not advised to be run on GPUs due to limited stack space
### 2) Only works for lists with less than 1024 elements
The benefits of parallel programs are greatest for large lists. Since moving data into GPU memory is expensive, serial algorithms are often faster at performing reduction tasks on smaller arrays.
### 3) Modifies the array in place
Running this implementation would modify the original ```nums``` array, which makes it useless in many contexts (e.g. sorting, etc.).