+++ 
date = "2018-11-18"
title = "What even is linear?"
markup = "mmark"
+++
 
 The list of engineering ideas centered on linearity is endless: linear equations, linear functions, linear systems, linear algebra, linear regression, linear filter, linear transforms, non-linear neural networks etc etc. We probably all think straight line when we think of the word linear. In fact, my first strong association of the word is from physics: the recti*linear* propagation of light. Light travels in a straight line. But what do straight lines have to do with all those listed ideas? Can we define linearity in a context independent manner? In this post, we’ll start with elementary math and build up a definition that can be applied whenever we encounter the word linear. My goal in doing this exercise was to replace the straight line mental picture with some other picture that makes more sense, and is more useful.


## Linear equations

An equation relates variables with equality and takes the following form for $n$ variables:

${\displaystyle a_{1}x_{1}+\cdots +a_{n}x_{n}+b=0}$


- The highest degree in this equation is 1. 

- Solving this equation refers to finding values for the variables that make the equation hold true. There can be more than one solution. The solution set contains all the solutions to the equation. When n = 2, the solution set is infintely large and contains pairs of values that live on a straight line. To see this geometrically, we can plot the pairs of variables in a two dimensional graph. Each axis represents a variable.

![linear_equation](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Linear_Function_Graph.svg/800px-Linear_Function_Graph.svg.png)


- The generalization of the solution set for $n > 2$ is a hyperplane (line in 2D plane, [a plane in 3D](https://www.wolframalpha.com/input/?i=x%2By%2Bz+%3D+0), a cube in 4D etc.). So linear does not always imply straight lines!

## Linear function

A function is an association of an element x in one set X to another element in set Y. It is most commonly specified as an equation:

$y = f(x)$

$f(x)$ is a linear function if the above equation is linear:

$${a_{1}f(x) + a_{2}x + b = 0}$$ or $${f(x) = (-a_{2}/a_{1})x-b}$$

All we have done so far is called one of the variables in a linear equation with $n=2$ as the "output" of a function. We often refer to the output of a function as the dependent variable and the "input" $x$ as the independent variable.

We can also generalize a linear function as having n independent variables:

${Y = f(x_{1}, x_{2}...x_{n}) = a_{1}x_{1} + \cdots +a_{n}x_{n} + b}$

This gets a little messy so we stack up $x_1...x_n$ into a list of numbers and call the new list object a vector. The vector is said to be n dimensional. 

## Linear algebra & system of equations

Algebra provides methods to solve equations and linear algebra to solve linear equations like the one in (1). It enables the solving of multiple equations at once i.e a system of equations:
${y_{1} = a_{1}x_{1} + \cdots +a_{n}x_{n} + b 
y_{2} = a_{1}x_{1} + \cdots +a_{n}x_{n} + b}$

We stack up the $y_{i}$s into a list and call it the $y$ vector. And replace the coefficients with a matrix $A$:

A system of equations is thus $y = Ax + b$. 

This is how linear algebra was introduced to me. The topic goes into great depth into the properties of $A$ which captures all the information regarding a linear function. But the system of equations picture is barely revelatory! Things really started to click when I was introduced to the notion of vector spaces defined using sets: 

### Vector space

### Define basis

With the ideas of vector spaces defined, we can re-interpret the system of equations as linear maps/transforms.

## Linear maps/tranformation

Linear maps are largely an interpretation of the system of equations y = Ax. A can be thought of as transforming an input vector x into the output vector y. 

Here is the definition of linearity we've been building up to: 
A is a linear map or function or transformation if the following properties hold (superposition) 


### Example: Fourier transforms as linear transforms

Fourer transforms takes points in the vector space of real numbers into the vector space of complex numbers. They are linear transforms 

### Example: Linear regression

### Example: Linear time invariant systems (convolution as a linear operation)

### Example: Linear dynamic systems

