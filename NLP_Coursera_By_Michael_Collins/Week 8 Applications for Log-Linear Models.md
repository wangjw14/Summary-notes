# Week 8 Applications for Log-Linear Models

## Log-Linear Models for Tagging （Maximum-entropy Markov Models (MEMMs)）

- Log-linear models for tagging
  - we have a input sentence $w_{[1;n]}=w_1,w_2,...,w_n$ ($w_1$ is the $i^\prime$ th word in the sentence)

  - we have a tag sequence $t_{[1:m]}=t_1,t_2,...,t_n$ ($t_i$ is the $i^\prime$ th tag in the sentence)

  - we use an log-linear model to define
    $$
    p(t_1,t_2,...,t_n|w_1,w_2,...,w_n)
    $$
    for any sentence $w_{[1:n]}$ and tag sequence $t_{[1:n]}$ of the same length.(**Note:** contrast with HMM that defines $p(t_1,...,t_n,w_1,...,w_n)$)

  - The most likely tag sequence for $w_{[1:n]}$ is

  $$
  t^*_{[1:n]} =\arg\max_{ t_{[1:n]}}p(t_{[1:n]}|w_{[1:n]})
  $$

- A trigram log-linear model tagger
  $$
  \begin{align}
  p(t_{[1:n]}|w_{[1:n]})&=\prod^n_{j=1}p(t_j|w_1...w_n,t_1...t_{j-1}) \ \ \ \ \text{ Chain rule} \\
  &=\prod^n_{j=1}p(t_j|w_1...w_n,t_{j-2},t_{j-1}) \ \ \ \ \text{ Independence assumptions} 
  \end{align}
  $$

  - We take $t_0=t_{-1}=*$

  - Independence assumption: each tag only depends on previous two tags
    $$
    p(t_j|w_1...w_n,t_1...t_{j-1})=p(t_j|w_1...w_n,t_{j-2},t_{j-1})
    $$

- Representation: Histories

  - A **history** is a 4-tuple $\langle t_{-2},t_{-1},w_{[1:n]},i \rangle$
  - $t_{-2},t_{-1}$ are the previous two tags
  - $w_{[1:n]}$ are the $n$ words in the input sentence
  - $i$ is the index of the word being tagged
  - $\mathcal X$ is the set of all possible histories

- Features can be used

  - Word/tag features for all word/tag pairs
  - Spelling features for all prefixes/suffixes of length $\le 4$
  - Contextual features

- Traininng the log-linear model

  - To train a log-linear model, we need a training set $x_i,y_i$ for $i=1...n$ Then search for
    $$
    v^*=\arg\max_v 
    \left(\underbrace{ \sum_i \log p(y_i|x_i;v)}_{\color{red}{Log-Likelihood}} - 
    \underbrace{\frac \lambda 2\sum_k v_k^2}_{\color{red}{Regularizer}}
    \right)
    $$

  - Training set is simply all history/tag pairs seen in the training data

- The Viterbi Algorithm

  - Some definition

    - $n$ is the length of the sentence

    - Define
      $$
      r(t_1...t_k)=\prod_{i=1}^kq(t_i|t_{i-2},t_{i-1},w_{[1:n]},i)
      $$

    - Define a dynamic programming table
      $$
      \begin{align}
      \pi(k,u,v)= &\text{ maximum probablity of a tag sequence ending} \\ &\text{ in tags $u,v$ at position}\ k
      \end{align}
      $$
      that is,
      $$
      \pi(k,u,v)= \max_{\langle t_1...t_{k-2} \rangle}r(t_1...t_{k-2},u,v)
      $$
      And base case is
      $$
      \pi(0,*,*)=1
      $$

    - Recursive definition for dynamic programming table

      For any $k\in\{1...n\}$, for any $u\in\mathcal{S}_{k-1}$ and $v\in\mathcal{S}_k$:
      $$
      \pi(k,u,v)=\max_{t\in \mathcal{S}_{k-2}} \left(\pi(k-1,t,u)\times q(v|t,u,w_{[1:n]},k) \right)
      $$
      where $\mathcal{S}_k$ is the set of possible tags at position $k$

  - The algorithm

    **Input:** a sentence $w_1...w_n$, a log-linear model that provide $q(v|t,u,w_{[1:n]},i)$ for any tag-trigram $t,u,v$, for any $i\in\{1...n\}$

    **Initialization:** Set $\pi(0,*,*)=1$

    **Algorithm:**

    - For $k=1...n$

      - For $u\in\mathcal{S}_{k-1},v\in\mathcal{S}_k$
        $$
        \pi(k,u,v)=\max_{t\in \mathcal{S}_{k-2}} \left(\pi(k-1,t,u)\times q(v|t,u,w_{[1:n]},k) \right)
        $$

        $$
        bp(k,u,v)= \arg\max_{t\in \mathcal{S}_{k-2}} \left(\pi(k-1,t,u)\times q(v|t,u,w_{[1:n]},k) \right)
        $$

    - Set $(t_{n-1},t_n)=\arg\max_{u,v}\pi(n,u,v)$
    - For $k=(n-2)...1$, $t_k=bp(k+2,t_{k+1},t_{k+2})$
    - **Return** the tag sequence $t_1...t_n$ 

- Summary

  - Key ideas in log-linear taggers

    - Decompose
      $$
      p(t_1...t_n|w_1...w_n)=\prod^n_{j=1}p(t_j|w_1...w_n,t_{j-2},t_{j-1})
      $$

    - Estimate
      $$
      p(t_j|w_1...w_n,t_{j-2},t_{j-1})
      $$
      using a log-linear model

    - For a test sentence $w_1...w_n$, use the Viterbi Algorithm to find
      $$
      \arg\max_{t_1...t_n} \left(\prod^n_{j=1}p(t_j|w_1...w_n,t_{j-2},t_{j-1}) \right)
      $$

  - Key advantages over HMM taggers: flexibility in the features they can use.



## Log-Linear Models for Parsing



