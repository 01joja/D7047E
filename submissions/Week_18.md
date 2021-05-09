# Week 18

Watched videos Attention and word2vec. 2,5 h

Only listened to Andrej Karpathys video. But that felt like a repetition of last week






## Notes word2vec video:

Discrete representation becomes big fast.

Example Discrete representation.

$$ \begin{bmatrix} 0, & 0, & 0, & 1, & 0  \end{bmatrix}= horse $$

One problem is that similarities between different words are hard to get. If we want hotel and motel we cant get them to impact each other.

$$ hotel \begin{bmatrix} 0, & 0, & 1, & 0, & 0  \end{bmatrix}^T motel \begin{bmatrix} 0, & 1, & 0, & 0, & 0  \end{bmatrix} = 0 $$



