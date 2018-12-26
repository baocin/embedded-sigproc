+++ 
date = "2018-11-18"
title = "What even is linear?"
markup = "mmark"
+++
 
 The list of engineering ideas centered on linearity is endless: linear regression, linear time-invariant (LTI) filter, linear dynamical system, non-linear neural networks etc. Most of these ideas are introduced with a cursorsy view of the significance of linearity. In this post, I want to flip the perspective and make linearity the central theme and elucidate how other ideas depend on it.
 
 We probably all think straight line when we think of the word linear. In fact, my first strong association of the word is from physics: the recti*linear* propagation of light i.e light travels in a straight line. But what do straight lines have to do with all those listed ideas? Can we define linearity in a context independent manner? We’ll start with elementary math and build up a definition that can be applied whenever we encounter the word linear. Ultimately, we'll use linearity in understanding Fourier transforms. My goal in doing this exercise was to replace the straight line mental picture with some other picture that makes more sense and is more useful.


## Linear equations

An equation relates variables with equality and takes the following form for $n$ variables:

${\displaystyle a_{1}x_{1}+\cdots +a_{n}x_{n}+b=0}$


- The highest [degree](https://en.wikipedia.org/wiki/Degree_of_a_polynomial) in this equation is 1. 

- Solving this equation refers to finding values for the variables that make the equation hold true. There can be more than one solution. The solution set contains all the solutions to the equation. When n = 2, the solution set is infintely large and contains pairs of values that live on a straight line. To see this geometrically, we can plot the pairs of variables in a two dimensional graph. Each axis represents a variable.

![linear_equation](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Linear_Function_Graph.svg/800px-Linear_Function_Graph.svg.png)


- The generalization of the solution set for $n > 2$ is a hyperplane (line in 2D plane, [a plane in 3D](https://www.wolframalpha.com/input/?i=x%2By%2Bz+%3D+0), a cube in 4D etc.). So linear does not always imply straight lines!

## Linear function

A function is an association of an element x in one set X to another element in set Y. It is most commonly specified as an equation:

$y = f(x)$

$f(x)$ is a linear function if the above equation is linear:

${a_{1}f(x) + a_{2}x + b = 0}$

or ${f(x) = (-a_{2}/a_{1})x-b}$

All we have done so far is called one of the variables in a linear equation with $n=2$ as the output of a function. We often refer to the output of a function as the dependent variable and the input $x$ as the independent variable.

We can also generalize a linear function as having n independent variables:

${Y = f(x_{1}, x_{2}...x_{n}) = a_{1}x_{1} + \cdots +a_{n}x_{n} + b}$

This gets a little messy so we stack up $x_1...x_n$ into a list of numbers and call the new list object a vector. The vector is said to be $$n$$ dimensional. 

## Linear algebra & system of equations

Algebra provides methods to solve equations and linear algebra to solve linear equations like the one in (1). It enables the solving of multiple equations at once i.e a system of equations:

${y_{1} = a_{11}x_{1} + \cdots +a_{1n}x_{n} + b}$ 

${y_{2} = a_{21}x_{1} + \cdots +a_{2n}x_{n} + b}$

${y_{3} = a_{31}x_{1} + \cdots +a_{3n}x_{n} + b}$

We stack up the $y_{i}$s into a list and call it the $y$ vector. And replace the coefficients with a matrix $A$:

A system of equations is thus $y = Ax + b$. 

This is how linear algebra was first introduced to me. The topic goes into great depth into the properties of the matrix $A$ which captures all the information regarding a linear function. Solving the system refers to finding all the values in the vector $$x$$ that satisfy the equation.

The system of equations picture is barely revelatory! It is one specific example of a more general idea for linearity.

## Linear map/tranformation/operation

An operation/transformation/map $$\mathbb{L}(x)$$ takes in an input vector $$x$$ and outputs/transforms/maps to a vector $$y$$. 

Now's the time for the definition of linearity we've been building up to. A map/transformation/operation is linear if the superposition property holds:

$\mathbb{F}(\alpha x_1 + \beta x_2) = \alpha \mathbb{F}(x_1) + \beta \mathbb{F}(x_2)$

The equation tells us that if we take two vectors $$x_1$$ and $$x_2$$, scale them with $$\alpha$$ and $$\beta$$ and pass the result through the linear operator then the output is the same as if we had passed each output separately, scaled them and then added the two results to produce the output.

In the system of equations picture of $$y = Ax$$, matrix A is a linear operator because it satisfies this superposition property. In fact, every linear function can be written as a matrix. 

Before we start applying this definition of linearity we need to paint a better picture for what a vector is. 

### Abstract vector space

We just saw how the system of equations picture gives us a notion for what a vector is: a list of numbers. Geometrically, we can think of the list as points in a "space". If the numbers are real numbers and the list has $$n$$ elements, the vector belongs to a vector space of n-dimensional real-valued points ${\Re^{n}}$. 

Now the interesting bit is that a list of numbers is only one example for what a vector can be. You can define other objects as vectors belonging to some vector "space". We will see examples in a second but before that, here is the general definition of what a vector space is.

A vector space consists of four objects:
- A set $$V$$
- A definition for a sum operation $$+$$ that takes in two elements from $$V$$ and outputs another element inside $$V$$
- A definition for a scalar multiplication operation $$x$$ that takes in a real number and one elements from $$V$$ and outputs another element inside $$V$$
- A special "zero" object $$0$$ inside $$V$$

These four objects together need to satisfy eight properties/axioms which are listed [here](http://mathworld.wolfram.com/VectorSpace.html). 

One object that qualifes as a vector is a function $$f(a)$$. We can have a vector space that consists of two functions $$f(a)$$ and $$g(a)$$ and the distinguised element $$0$$ and define operations of addition ($$(f+g)(a)$$) and multiplication ($$(f*g)(a)$$) that satisfy the eight axioms. As a more specific example, we can define the vector space of polynomials of degree 2 as a set that contains the three functions $$f(x) = 1$$, $$g(x) = x$$ and $$h(x) = x^2$$. Differentiation $$d/dx$$ is an example of a linear operator that can act on this vector space of polynomials. We can re-write differentiation in a matrix form D as well! This brilliant 3blue1brown [video](https://www.youtube.com/watch?v=TgKwz5Ikpc8) captures how.

As an interesting side, because random variables are functions (and hence vectors) too, ideas from linear algebra are easily applied to statistics as well. [Here's](https://www.randomservices.org/random/expect/Spaces.html) more info. 


### Basis of a vector space

The basis of a vector space is a set of vectors that can be thought of as "building blocks" of the entire vector space. 

### Example: Fourier transforms as linear transforms

Fourier transforms are ubiquitous in engineering. They allow us to break down any signal/function into a summation of different sinusoid functions of varying amplitudes and phases. The equation for the transformation looks like:
 
$\mathscr{F}\{f(t)\}=F(\omega)$

$\mathscr{F}\{f(t)\}=\int_{-\infty}^{\infty} f(t) e^{-i \omega t} dt$

This is a scary equation at first but with the idea of vector spaces and linearity, we can re-interpret this quickly. 

- The Fourier transform is a linear mapping because the superposition property holds. 
- It maps a function to another function. The input function is often interpreted as a function of time $$t$$ and the output as a function of frequency $$\omega$$. The input function $$f(t)$$ comes from the vector space of functions of real numbers and $$f(t)$$ itself maps a real number to another real number. The output function $$F(\omega)$$ belongs to the vector space of complex valued functions. $$F(\omega)$$ takes in a real valued frequency and outputs a complex value.
- Complex numbers are used because a single complex number captures all the information about a sinusoid. Any complex number can be written in phasor form as $$Ae^{i\phi}$$ where $$A$$ can be thought of as the amplitude of the sinusoid and $$\phi$$ as its phase.

- The integral is probably the scariest bit. To understand thWhat it's doing is an inner product between the 