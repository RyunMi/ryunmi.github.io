---
title: Paper Reading 1 - Learning Laplacian Matrix from Graph Signals with Sparse Spectral Representation
date: 2022-09-25 18:00:00
categories:
- Paper Reading
tags:
- Optimization
- Graph learning
# cover: /images/posts/read1.png
# banner: /images/posts/read1.png
---

<img src="/images/posts/read1.png" style="display: block; margin: auto;" />

# Problem introduction
---
The paper introduced an optimization problem to learn a graph from signals that are assumed to be smooth and admitting a sparse representation in the spectral domain of the graph.

<!--more-->

> *What is graph learning*
> Considering an undirected and weighted graph G=($\mathcal{V}$,$\mathcal{E}$) with no self-loops, where $\mathcal{V}=\lbrace 1,\dots,N \rbrace$ represent the node and $\mathcal{E}=\lbrace (i, j, w_{i j}), i, j \in \mathcal{V}\rbrace$ represent the edge with weights $w_{i j}$.
> The paper focus on learning the Laplacian matrix $L$(Symmetric positive semi-define matrix)$=D - W$, where $D$ is the diagonal degree matrix and $W$ the weight matrix. $L's$ eigenvalue decomposition is $L=X \Lambda X^{\top}$ with $\Lambda=\operatorname{diag}\left(\lambda_1, \ldots, \lambda_N\right)$ a diagonal matrix with the eigenvalues and $X=\left(x_1, \ldots, x_N\right)$ a matrix with the eigenvectors as columns.
> A graph signal is defined as a function $y: \mathcal{V} \rightarrow \mathbb{R}^N$ that assigns a scalar value to each vertex.
The two concepts of smoothness and spectral representation in graph learning problems are explained as follows:

>**Smoothness**
>A graph signal $y$ is s-smooth with respect to the graph:\begin{equation} y^{\top} L y=\frac{1}{2} \sum_{i, j} w_{i j}\left(y_i-y_j\right)^2 \leq s\tag{1} \label{smooth} \end{equation} where $s\ge 0$ is smoothness level.

>**Spectral sparsity**
>A graph signal $y$ admits a $k$-sparse spectral representation with respect to the graph: $$\|h\|_0 \leq k$$ where $h=X^Ty$ for Graph Fourier Transform.

From an intuitive point of view, smoothness means that the signal values between adjacent nodes are close, and k-sparse spectral representation means that in the overall graph, it can be divided into k clusters, and the signal values are smooth within the cluster and vary greatly between clusters.

# Abstract optimization problem
---
$$\min _{H, X, \Lambda}\|Y-X H\|_F^2+\alpha\left\|\Lambda^{1 / 2} H\right\|_F^2+\beta\|H\|_S,\\\\
s.t. \begin{cases}
X^{\top} X=I_N, x_1=\frac{1}{\sqrt{N}} \mathbf{1}_N, \text { (a) } \\\\
(X \Lambda X^{\top})_{k,\ell} \leq 0 \quad k \neq \ell, \text { (b) } \\\\
\Lambda=\operatorname{diag} (0, \lambda_2, \ldots, \lambda_N) \succeq 0, \text { (c) } \\\\
\operatorname{tr}(\Lambda)=N \in \mathbb{R}_{*}^{+} \text { (d) } \\\\
\end{cases}
$$
+ $\|Y-X H\|_F^2$ controls the distance of the new representation $XH$ to the observations $Y$.
+ From equation $(1)$, $\sum_i y^{(i) \top} L y^{(i)}=\operatorname{tr}\left(Y^{\top} L Y\right)=\left\|L^{1 / 2} Y\right\|_F^2$, then replacing $Y$ with $XH$, obtaing $\left\|L^{1 / 2} X H\right\|_F^2=\left\|X \Lambda^{1 / 2} X^{\top} X H\right\|_F^2=\left\|X\right\|_F^2 \left\|\Lambda^{1 / 2}  H\right\|_F^2=\operatorname{tr}(X^TX)\left\|\Lambda^{1 / 2} H\right\|_F^2=\left\|\Lambda^{1 / 2} H\right\|_F^2$. This objective item controls the smoothness of the new representation.
+ $\beta\|H\|\_S$ is a sparsity regularization (as $h=X^Ty=X^{-1}y\Rightarrow Xh=y \Rightarrow XH \rightarrow Y$). The $\|\cdot\|\_S$ here is often expressed as $\mathcal{l}\_\{1,2\}$ (the sum of the $\mathcal{l}_\{2\}$-norm of each row of $H$) in the paper.
+ Constraints (a), (b), (c) originate from the properties of Laplacian matrix $L$.
+ Constraint (d) was as to impose structure in the learned graph while avoiding that the trivial solution $\widehat{\Lambda}=0$.

# Reformulation of the problem
---
>**Reformulation of the constraint $(a)$**
>Given $X, X_0 \in \mathbb{R}^{N \times N}$ two orthogonal matrices, both having their first column equal to $\frac{1}{\sqrt{N}}\mathbf{1}\_N$. And $$ X=X_0 \left[\begin{array}{cc} 1 & \mathbf{0}^{\top}_{N-1} \\\\ \mathbf{0}_{N-1} & [X_0^{\top}X]^{}_{2:,2:} \end{array}\right] $$ where $[X_0^\{\top\} X]_\{2:, 2:\} := U $ is in $\operatorname{Orth}(N-1)=\lbrace X \in \mathbb{R}^\{N-1 \times N-1\} \mid X^\{\top\} X=I_N \rbrace$.

Constraint (a) is still satisfied after the above reformulation, and the optimization problem is transformed into the following:
$$\min_{H, U, \Lambda}\left\|Y-X_0\left[\begin{array}{cc}1 & \mathbf{0}^{\top}_{N-1} \\\\ \mathbf{0}_{N-1} & U\end{array}\right] H\right\|_F^2+\alpha\left\|\Lambda^{1 / 2} H\right\|_F^2+\beta\|H\|_S \triangleq f(H, U, \Lambda) \\\\
s.t. \begin{cases}
U^{\top} U=I_{N-1}, \quad ( a' )\\\\ 
\left(X_0\left[\begin{array}{cc}1 & \mathbf{0}^{\top}_{N-1} \\\\
\mathbf{0}_{N-1} & U\end{array}\right] \Lambda\left[\begin{array}{cc}1 & \mathbf{0}^{\top}_{N-1} \\\\
\mathbf{0}_{N-1} & U^{\top}\end{array}\right] X_0^{\top}\right)_{k, \ell} \leq 0 \quad k \neq \ell, \quad ( b^{\prime} )\\\\
\Lambda=\operatorname{diag}\left(0, \lambda_2, \ldots, \lambda_N\right) \succeq 0, \text { (c) } \\\\
\operatorname{tr}(\Lambda)=N \in \mathbb{R}^{+}_{*} \text { (d) }
\end{cases}$$
Further, the paper uses the log-barrier method to deal with the constraint ($b^{\prime}$), as follows:
>**Reformulation of the constraint $(b^{\prime})$**
> $$ \min_{H, U, \Lambda} f(H, U, \Lambda)-\frac{1}{t}\sum^{N-1}_{k=1} \sum_{\ell>k}^N \log \left(-h(U, \Lambda)_{k, \ell}\right) \quad \text{ s.t. } \quad\left( a^{\prime}\right),( \mathrm{c}),(\mathrm{d}) $$ 
where $ h(U, \Lambda)=X_0 \left[\begin{array}{cc}1 & \mathbf{0}^{\top}\_{N-1\} \\\\ \mathbf{0}\_{N-1} & U\end{array}\right] \Lambda\left[\begin{array}{cc}1 & \mathbf{0}^{\top}\_{N-1\} \\\\ \mathbf{0}\_{N-1\} & U\end{array}\right]^{\top} X_0^\{\top\}$,
> with $\operatorname{dom}(\phi)=\lbrace (U, \Lambda) \in \mathbb{R}^\{(N-1) \times(N-1)\} \times \mathbb{R}^\{N \times N\} \mid \forall 1 \leq k<\ell \leq N, h(U, \Lambda)_\{k, \ell\}<0 \rbrace$.

$\operatorname{dom}(\phi)$ just satisfies the constraint ($b^{\prime}$), and the paper mentioned that "This barrier function allows us to perform block-coordinate descent on three subproblems
that are easier to solve".

# IGL-3SR and FGL-3SR
---
## Iterative Graph Learning for Smooth and Sparse Spectral Representation
Spliting the problem in three partial minimizations and using a block-coordinate descent on $H$, $U$, and $\Lambda$.
Input:$ Y \in \mathbb{R}^\{N \times n\},\alpha,\beta$ and $t^\{(0)\},t_\{max\},\mu$ based on path-following method.
> **Optimization with respect to $H$**
> For fixed $U$ and $\Lambda$, the minimization problem with respect to $H$ is: $$ \min_H\|Y-X H\|_F^2+\alpha\left\|\Lambda^{1 / 2} H\right\|_F^2+\beta\|H\|_S, \quad \text{where} X=X_0\left[\begin{array}{cc} 1 & \mathbf{0}^{\top}_{N-1} \\\\ \mathbf{0}_{N-1} & U \end{array}\right] $$ The solution of problem above when $\|\cdot\|\_S$ is set to $\|\cdot\|\_{2,1}$ is: $$ \widehat{H}_{i,:}=\frac{1}{1+\alpha \lambda_i}\max\left(0,1-\frac{\beta}{2} \frac{1}{\left\|\left(X^{\top} Y\right)_{i,:}\right\|_2}\right)\left(X^{\top} Y\right)_{i,:} $$

> **Optimization with respect to $\Lambda$**
> For fixed $H$ and $U$, the minimization problem with respect to $\Lambda$ is: $$ \min_{\Lambda} \alpha \underbrace{\operatorname{tr}\left(H H^{\top} \Lambda\right)}_{\left\|\Lambda^{1 / 2} H\right\|_F^2}+\frac{1}{t} \phi(U, \Lambda) \quad \text{ s.t. } \begin{cases} \Lambda=\operatorname{diag}\left(0, \lambda_2, \ldots, \lambda_N\right) \succeq 0 \\\\ \operatorname{tr}(\Lambda)=N \in \mathbb{R}^{+}_{*} \end{cases} $$ Methods: CVXPY solver, interior-point, projected gradient descent.

> **Optimization with respect to $U$**
> For fixed $H$ and $\Lambda$, the minimization problem with respect to $U$ is:$$ \min_U\left\|Y-X_0\left[\begin{array}{cc} 1 & \mathbf{0}^{\top}_{N-1} \\\\ \mathbf{0}_{N-1} & U\end{array}\right] H\right\|_F^2+\frac{1}{t} \phi(U, \Lambda) \quad \text{ s.t.} \quad U^{\top} U=I_{(N-1)} $$ Approaching the solution: $$U\leftarrow retraction(U([(HY^TX_0)_{2:,2:}]U-U^T[(HY^TX_0)_{2:,2:}]^T))$$

Considering the subproblem on $U$, the "retraction" represents a gradient descent method generalized to manifold--"consists in selecting, at each iteration, a search direction belonging to the tangent space of the manifold defined at the current point X, and then performing a descent along a curve of the manifold".
On the convergence of IGL-3SR, any accumulation point $\left(H^\{\infty\},U^\{\infty\},\Lambda^\{\infty\}\right)$ of the sequence generated by IGL-3SR($\lbrace\left(H^\{(\ell)\}, U^\{(\ell)\},\Lambda^\{(\ell)\}\right)\rbrace_\{\ell \geq 0\}$) satisfies the KKT conditions of problem. The paper proves that $H^\{\infty\},U^\{\infty\},\Lambda^\{\infty\}$ all satisfy the KKT conditions of their respective sub-problems, and the Bolazano-Weierstrass theorem can obtain a sequence that must converge to $(H^\{\infty\},U^\{\infty\},\Lambda^\{\infty\})$. The paper cleverly sets three sub-problems for three sequence, all converge to $(H^\{\infty\},U^\{\infty\},\Lambda^\{\infty\})$, indicating that $(H^\{\infty\},U^\{\infty\},\Lambda^\{\infty\})$ is a KKT solution.

## Fast Graph Learning for Smooth and Sparse Spectral Representation
Relying on a simplification of the minimization step in $X$ by removing the constraint (b).
The method of solving $H$ in FGL-3SR is the same as that in IGL-3SR. And ignoring the constraint (b) at the $X-step$ is to compute a closed-form solution to the optimization problem. Then using the learned X to optimize with respect to $\Lambda$.

> **Optimization with respect to $X$**
> During the X-step: $$\min _X\|Y-X H\|_F^2 \quad s.t.\quad X^{\top} X=I_N, x_1=\frac{1}{\sqrt{N}} \mathbf{1}_N$$ Solving the X:
>$M \leftarrow\left(X^{\top} Y H^{\top}\right)\_{2:, 2:}, \left(P, D, Q^{\top}\right) \leftarrow \operatorname{SVD}(M), X \leftarrow X\left[\begin{array}{cc}1 & \mathbf{0}^{\top}\_{N-1} \\\\ \mathbf{0}\_{N-1} & P Q^{\top}\end{array}\right]$.

> **Optimization with respect to $\Lambda$**
> $$\min_{\Lambda} \alpha \underbrace{\operatorname{tr}\left(H H^{\top} \Lambda\right)}_{\left\|\Lambda^{1 / 2} H\right\|_F^2} \quad s.t. \begin{cases} \left(X \Lambda X^{\top}\right)_{i, j} \leq 0 \quad i \neq j, \\\\ \Lambda=\operatorname{diag}\left(0, \lambda_2, \ldots, \lambda_N\right) \succeq 0, \\\\ \operatorname{tr}(\Lambda)=N \in \mathbb{R}^{+}_{*}.\end{cases}.$$ Methods: interior-point, ellipsoid method.

After the complexity analysis of the paper, the conclusion of "the empirical execution time of FGL-3SR is lower than IGL-3SR" is obtained.

# Numerical simulation and others
---
Subsequently, the paper provide a probabilistic interpretation of the optimization program and test the algorithms and comparing it to state-of-the-art approaches on several synthetic and real datasets.
As stated in the paper "The findings of our empirical evaluation on synthetic data showed that the proposed approaches are as good or better performing than the reference state-of-the-art algorithms in term of reconstructing the unknown underlying graph and of computational cost (running time). Experiments on real-world benchmark use-cases suggest that our algorithms learn graphs that are useful and promising for any graph-based machine learning methodology, such as graph clustering and subsampling, etc" demonstrate the excellent performance of both algorithms.

---
[Original paper link](https://www.jmlr.org/papers/v22/19-944.html)