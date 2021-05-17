# Theory Exercise 4

## 1.1 Task 

### 1.1.1 Question

For 5 timesteps

$$
(20 \cdot 128 + 128 \cdot 10 )\cdot 5 + 128 \cdot 4 = 19712.
$$

For 10 timesteps

$$
(20 \cdot 128 + 128 \cdot 10 )\cdot 10 + 128 \cdot 9 = 39552.
$$

### 1.1.2  Question 2

No. The parameters are the same during a forward roll they are only used more times depending on the number of timesetps you are taking. During a backward roll the parameter will change for every timestep but no new parameters are added.

## 1.2 Task

### 1.2.1  Question 1

The vanishing or exploding gradient problem. Because you are multiplying with the same scalar every time step the value will vanish or explode if the scalar is not equal to 1.

### 1.2.2  Question 2

LSTM 

GRU
