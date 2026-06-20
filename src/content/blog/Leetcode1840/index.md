---
title: 'Leetcode 每日一题 1840. 最高建筑高度'
publishDate: 2026-06-20
updatedDate: 2026-06-20
description: ‘Leetcode 每日一题'
tags:
  - Leetcode
  - 算法题
language: '中文'
heroImage: { src: './image.png', color: '#D58388' }
---



在一座城市里，你需要建 n 栋新的建筑。这些新的建筑会从 1 到 n 编号排成一列。

这座城市对这些新建筑有一些规定：

- 每栋建筑的高度必须是一个非负整数。
- 第一栋建筑的高度 必须 是 0 。
- 任意两栋相邻建筑的高度差 不能超过  1 。

除此以外，某些建筑还有额外的最高高度限制。这些限制会以二维整数数组 restrictions 的形式给出，其中 $restrictions[i] = [id_i, maxHeight_i]$ ，表示建筑 $id_i$ 的高度 不能超过 $maxHeight_i$ 。

题目保证每栋建筑在 restrictions 中 至多出现一次 ，同时建筑 1 不会 出现在 restrictions 中。

请你返回 最高 建筑能达到的 最高高度 


> 输入：n = 5, restrictions = \[[2,1],[4,1]]
> 输出：2



- $2 <= n <= 10^9$
- $0 <= restrictions.length <= min(n - 1, 10^5)$
- $2 <= id_i <= n$
- $id_i$ 是 唯一的 。
- $0 <= maxHeight_i <= 10^9$



