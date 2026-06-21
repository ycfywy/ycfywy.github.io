---
title: 'Dijkstra算法及其变体'
publishDate: 2026-06-21
updatedDate: 2026-06-21
description: ‘最短路算法 Dijkstra 很常用，本文章讲解最常用的堆优化版本的Dijkstra算法及其变体（维护除距离之外的内容）'
tags:
  - 算法
heroImage: { src: './image.png', color: '#839dd5' }
---


### Dijkstra 算法核心思想

Dijkstra 算法的核心思想可以总结为四个字：贪心与松弛。

想象一下，你被困在一个由迷雾笼罩的复杂迷宫中，每个房间（节点）之间有不同长度的通道。你想知道从你所在的起点房间出发，到迷宫中其他所有房间的最短距离。你手里有一张距离表，初始时除了起点写着 0 之外，其他所有房间的距离都填着“无穷大”。我们会不断地执行松弛操作，直到dist无法再更新。



$$\text{If } dist[u] + w(u, v) < dist[v] \implies dist[v] = dist[u] + w(u, v)$$

### 堆优化版的Dijkstra

```cpp

#include <iostream>
#include <vector>
#include <queue>

using namespace std;

const int INF = 0x3f3f3f3f; 
using Edge = pair<int, int>; 
// Graph 结构：邻接表，graph[u] 里面存了 u 能直达的所有 {v, weight}
using Graph = vector<vector<Edge>>;

vector<int> dijkstra(int n, const Graph& graph, int start) {

    vector<int> dist(n, INF);
    dist[start] = 0;

    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;

    pq.push({0, start});

    while(pq.size()){
        auto t = pq.top();
        pq.pop();
        int d = t.first, u = t.second;
        
        // 剪枝：如果弹出的距离已经比记录的更长，说明是过期的失效路径
        if (d > dist[u]) continue;
        for(auto& [v, w]: graph[u]){
            if(dist[v] > d + w){
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;

}

```



### 限制总开销的最短路（状态扩展 Dijkstra）

每条边带有 weight（距离）和 cost（开销）。要求在总开销不超过 max_cost 的前提下，求最短距离。

```cpp

#include <iostream>
#include <vector>
#include <queue>
#include <tuple>

using namespace std;

// 定义图的边结构
struct Edge {
    int to;
    int weight;
    int cost;
};

// 优先队列中存储的状态：{当前总距离, 当前节点, 当前总开销}
// C++ 的 tuple 默认会按第一个元素（distance）从小到大排序
using State = tuple<int, int, int>;

int constrainedDijkstra(const vector<vector<Edge>>& graph, int n, int start, int end, int max_cost) {
    // dist[u][c] 表示到达节点 u 且开销刚好为 c 时的最短距离
    // 初始化为无穷大
    const int INF = 1e9;
    vector<vector<int>> dist(n, vector<int>(max_cost + 1, INF));
    
    // 小顶堆优先队列
    priority_queue<State, vector<State>, greater<State>> pq;
    
    // 起点初始化
    dist[start][0] = 0;
    pq.push({0, start, 0});
    
    while (!pq.empty()) {
        auto [d, u, c] = pq.top();
        pq.pop();
        
        // 第一次弹出终点时，由于优先队列的贪心特性，必然是合法开销下的最短路
        if (u == end) {
            return d;
        }
        
        // 剪枝
        if (d > dist[u][c]) continue;
        
        // 遍历邻居
        for (const auto& edge : graph[u]) {
            int next_cost = c + edge.cost;
            
            // 约束拦截：总开销不能超过上限
            if (next_cost <= max_cost) {
                if (d + edge.weight < dist[edge.to][next_cost]) {
                    dist[edge.to][next_cost] = d + edge.weight;
                    pq.push({dist[edge.to][next_cost], edge.to, next_cost});
                }
            }
        }
    }
    
    return -1; // 无法在约束条件下到达终点
}





```


证明“状态扩展 Dijkstra”（或称分层图 Dijkstra）的正确性，核心思想非常美妙：**它本质上并没有修改 Dijkstra 算法本身，而是通过“转生”拓扑结构，把一个“有约束的复杂问题”，降维成了一个“无约束的标准最短路问题”。**

要严谨地证明它的正确性，我们可以分三个层层递进的步骤来推导：

---

## 1. 核心：状态空间图的等价转换（Reduction）

标准的 Dijkstra 算法之所以正确，是因为它运行在一个满足“边权非负”的图 $G = (V, E)$ 上。

当我们引入约束（最大开销 $C_{max}$）后，我们其实是在隐式地构建一个全新的**状态空间图 (State Space Graph)**，记为 $G' = (V', E')$：

* **全新的节点集 $V'$**：原图中的一个点 $u$，在 $G'$ 中被分裂成了 $C_{max} + 1$ 个点。每一个新节点都是一个二元组 $(u, c)$，代表“到达物理节点 $u$ 且当前累计开销为 $c$”这一特定状态。
* **全新的边集 $E'$**：如果原图中存在一条边 $u \to v$，其距离为 $w$，开销为 $\text{cost}$。那么在状态图 $G'$ 中，就存在一条从 $(u, c)$ 指向 $(v, c + \text{cost})$ 的单向边，其权重同样为 $w$。
* **边界约束拦截**：如果 $c + \text{cost} > C_{max}$，这条边在 $G'$ 中直接**不存在**。

通过这种转换，原图中的“带约束路径”，就完美等价地映射成了新图 $G'$ 中的“无约束普通路径”。

---

## 2. 数学归纳法证明（基于 Dijkstra 贪心策略）

既然问题被等价转换为了新图 $G'$ 上的普通最短路，我们只需要证明**小顶堆中每次弹出的状态，都已经找到了到达该状态的绝对最短距离**。

我们用数学归纳法来证明这个贪心选择性：

### 初始状态

起点状态为 $(start, 0)$，距离为 0。显然，它是所有合法状态中距离最短的（边权非负），第一个被弹出，正确性成立。

### 假设与递推

假设目前已经有若干个状态被确认并弹出了，它们的 `dist[u][c]` 都是绝对正确的。现在，轮到状态 $(v, c_{next})$ 被从堆顶弹出。

**我们要证明：此时堆顶的 `d = dist[v][c_{next}]` 就是从起点到该状态的最短路径。**

**反证法**：
假设 `d` 不是最短的，这意味着在 $G'$ 中存在另一条更短的隐秘路径，能够以更小的距离 $d'$ 抵达 $(v, c_{next})$。

由于这条更短的路径最终也要走到 $(v, c_{next})$，它在 $G'$ 中一定存在某一个临界点 $(u, c_{current})$，该临界点是已经确认的最短路集合与未确认集合的交界处：

$$\text{起点} \longrightarrow (u, c_{current}) \overset{\text{单步边 } w'}{\longrightarrow} (v', c') \longrightarrow \dots \longrightarrow (v, c_{next})$$

根据我们的假设：

1. 因为边权非负（距离 $w \ge 0$），所以：

$$\text{距离}((u, c_{current})) + w' \le d'$$


2. 又因为我们假设这条隐秘路径更短，所以：

$$d' < d$$



把这两个不等式连起来，可以得到：


$$\text{距离}((u, c_{current})) + w' < d$$

这说明什么？说明在当前这一轮中，由 $(u, c_{current})$ 扩展出来的邻居 $(v', c')$ 的估计距离，**绝对比当前堆顶的 $d$ 还要小**！

按照小顶堆的性质，这个更小的状态 $(v', c')$ 应该**排在当前堆顶的前面、更早被弹出**才对。这与“当前弹出的是 $d$”这一事实产生了不可调和的矛盾。

因此，假设不成立。每次从优先队列顶端弹出的状态，其记录的距离必然是无法被进一步更新的绝对最短路。

---

## 3. 为什么传统的剪枝不会漏掉正确答案？

有人会担心：**如果一个点 $u$ 先以较大的距离、但极低的开销被访问了；后面又有一个较小的距离、但较高的开销想访问 $u$，传统的 Dijkstra 会不会把它剪枝掉？**

这正是状态扩展最精妙的地方：因为我们的状态是 `dist[u][c]`。

* 距离大、开销低的状态是 `dist[u][10] = 100`。
* 距离小、开销高的状态是 `dist[u][50] = 20`。

在二维数组中，它们存储在**不同的格子**里！在小顶堆里，它们是**两个独立的节点**。当距离小的状态去更新 `dist[u][50]` 时，由于 `20 < inf`，它会成功写入并入队，**根本不会触发对彼此的剪枝**。

只有当两个状态的 `node` 相同、且 `cost` 完美相同时（即走到了状态空间图的同一个分身点），由于开销一样，此时谁的距离大谁就是纯纯的垃圾解，才会触发传统的 `if (d > dist[u][c]) continue;` 剪枝。

---

## 📝 最终结论

有约束的最短路之所以一定正确，可以用一句话概括：

> **“它在数学上将原图的一维节点 $u$，无损重构成了解空间里的二维节点 $(u, c)$。在这个全新的非负权图上，标准 Dijkstra 算法的贪心正确性定理依然神圣不可侵犯。”**·


### 分层图最短路（K次免单/魔改机会）允
许最多将 $K$ 条边的权重直接清零（相当于跨越分层图）。求从起点到终点的最短路径。

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>

using namespace std;

struct SimpleEdge {
    int to;
    int weight;
};

// 优先队列状态：{当前总距离, 当前节点, 已使用的特权次数 K}
using LayerState = tuple<int, int, int>;

int layeredDijkstra(const vector<vector<SimpleEdge>>& graph, int n, int start, int end, int K) {
    const int INF = 1e9;
    // dist[u][k] 表示到达节点 u，使用了 k 次免单权时的最短距离
    vector<vector<int>> dist(n, vector<int>(K + 1, INF));
    
    priority_queue<LayerState, vector<LayerState>, greater<LayerState>> pq;
    
    dist[start][0] = 0;
    pq.push({0, start, 0});
    
    while (!pq.empty()) {
        auto [d, u, k] = pq.top();
        pq.pop();
        
        // ⚠️ 注意：分层图中终点可能在任何一层被触发，所以我们可以直接在弹出 end 时返回，
        // 或者等队列跑完去 min(dist[end][0...K])。这里遇到 end 直接返回也是安全的。
        if (u == end) {
            return d;
        }
        
        if (d > dist[u][k]) continue;
        
        for (const auto& edge : graph[u]) {
            // 选择 1：正常走，不使用特权，留在当前层（k 不变）
            if (d + edge.weight < dist[edge.to][k]) {
                dist[edge.to][k] = d + edge.weight;
                pq.push({dist[edge.to][k], edge.to, k});
            }
            
            // 选择 2：使用特权，边权算 0，跨越到下一层（k + 1）
            if (k < K) {
                if (d < dist[edge.to][k + 1]) { // d + 0
                    dist[edge.to][k + 1] = d;
                    pq.push({dist[edge.to][k + 1], edge.to, k + 1});
                }
            }
        }
    }
    
    // 如果没有在循环中直接返回，兜底寻找所有层中到达 end 的最小值
    int min_dist = INF;
    for (int k = 0; k <= K; ++k) {
        min_dist = min(min_dist, dist[end][k]);
    }
    
    return (min_dist == INF) ? -1 : min_dist;
}