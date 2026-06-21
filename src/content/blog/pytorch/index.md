---
title: 'Pytorch API 相关'
publishDate: 2026-06-21
updatedDate: 2026-06-21
description: ‘介绍各种Pytorch 常用API的使用方法及原理'
tags:
  - Pytorch
heroImage: { src: './image.png', color: '#839dd5' }
---



## Contiguous

在Pytorch中，contiguous 指的是Tensor的实际存储顺序与逻辑顺序是否一致。在讲解contiguous之前，我们有必要了解一下实际存储顺序与逻辑顺序。

### Tensor实际存储顺序

Pytorch的底层是C实现的，因此其存储顺序也和C语言一样，是行优先的。如下图所示，Tensor的存储是行优先存储的。对于更高纬度的tensor，比如（2， 3， 4）我们也是行优先，从最后一个维度开始，最后一个维度的所有元素在同一行。
```python
t = torch.arange(12).reshape(3,4)

t
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```
<p align="center">
  <img src="./row.png" />
</p>

### Tensor的逻辑顺序

当你创建了一个 Tensor，它在物理内存上是一条线（一维连续数组），但在逻辑上它是高维的。这全靠以下 2 个变量来完成“从一维到高维”的逻辑映射：

- shape: 举个例子，shape 为 (2, 3, 4)，它规定了逻辑上有 3 个轴（Axes），每个轴能容纳多少个元素。
- strides: 表示在逻辑上的某个维度前进 1 步，在物理内存中需要跨过多少个元素。比如(2, 3, 4) Tensor，其 stride 是 (12, 4, 1)。strides大小取决于当前位置往后的所有维度的乘积，好比 3 * 4
  

我们经常会使用一些 API 来改变tensor的shape，具体来说他们也有一些差异：transpose 和 permute并不会改变tensor的实际存储顺序，只会交换tensor的shape和strides，因此会导致tensor的存储顺序与逻辑顺序不一致，造成not contiguous。举个例子，你对上面的tensor做一个转置，那么tensor的顺序就变成了\[[0, 4, 8], [1, 5, 9], ....], 显然0， 4， 8的顺序是不连续的（访问的元素并不是连续存储）。而view操作则是重新划分维度，比如将(3， 4)的tensor视为(2, 6)。该操作要求tensor必须连续，他会重新设置新的strides。reshape操作比较万金油, 在 not contiguous 的情况下，他会直接调用contiguous整理tensor的存储位置，使之连续。

```python
transpose(dim0, dim1)
permute(*dims)
view(*shape)
reshape(*shape)
```
### 总结
contiguous 其实就是 pytorch 对tensor的实际存储与逻辑存储之间关系的一个设定，之所以会有该问题，是因为pytorch在某些API中，为了效率考虑，没有真正修改tensor实际存储状态。我们在使用相关API的时候要注意，在not contigusouu的情况下，调用contiguous()重新排列tensor。