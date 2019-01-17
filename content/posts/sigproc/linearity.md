+++ 
date = "2018-12-31"
title = "On linearity: straight lines, linear operators & the Fourier transform"
markup = "mmark"
+++
 
 The list of engineering ideas centered on linearity is endless: linear regression, linear time-invariant (LTI) filter, linear dynamical system, linearization, support vector machines, circuit theory/network analysis etc. Most of these ideas are introduced with a cursorsy view of the significance of linearity. In this post, I want to flip the perspective and make linearity the central theme and elucidate how other ideas depend on it.
 
 We probably all think straight line when we think of the word linear. In fact, my first strong association of the word is from physics: the recti*linear* propagation of light i.e light travels in a straight line. But what do straight lines have to do with all those listed ideas? Can we define linearity in a more generalized manner? We’ll start with elementary math and build up a definition that can be applied whenever we encounter the word linear. Ultimately, we'll use the new perspective of linearity in understanding a popular linear operator: the Fourier transform. My goal in doing this exercise was to replace the straight line mental picture with a more generalized notion of linearity.

## Linear equations: From straight lines to hyperplanes

An equation relates variables with equality ($=$). It takes the following form for $n$ variables:

${\displaystyle a_{1}x_{1}+\cdots +a_{n}x_{n}+b=0}$


- The highest [degree](https://en.wikipedia.org/wiki/Degree_of_a_polynomial) in this equation is 1. 

- Solving this equation refers to finding values for the variables that make the equation hold true. There can be more than one solution. The solution set contains all the solutions to the equation. When n = 2, the solution set is infintely large and contains pairs of values that live on a straight line. To see this geometrically, we can plot the pairs of variables in a two dimensional graph. Each axis represents a variable.

![linear_equation](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Linear_Function_Graph.svg/800px-Linear_Function_Graph.svg.png)


- The generalization of the solution set for $n > 2$ is a hyperplane (line in 2D plane, [a plane in 3D](https://www.wolframalpha.com/input/?i=x%2By%2Bz+%3D+0), a cube in 4D etc.). Hyperplanes are generalizations of straight lines in higher dimensional spaces.

### Linear function: Defining functions using equations

A function is an association of an element $$x$$ in one set $$X$$ to another element in set $$Y$$. It is most commonly specified as an equation:

$y = f(x)$

$f(x)$ is a linear function if the above equation is linear:

${a_{1}f(x) + a_{2}x + b = 0}$

or ${f(x) = (-a_{2}/a_{1})x-b}$

All we have done so far is called one of the variables in a linear equation with $n=2$ as the output of a function. We often refer to the output of a function as the dependent variable and the input $x$ as the independent variable.

We can also generalize a linear function as having n independent variables:

${Y = f(x_{1}, x_{2}...x_{n}) = a_{1}x_{1} + \cdots +a_{n}x_{n} + b}$

This gets a little messy so we stack up $x_1...x_n$ into a list of numbers and call the new list object a vector. The vector is said to be $$n$$ dimensional. 

### System of equations: Dealing with many variables

Algebra provides methods to solve equations and linear algebra to solve linear equations like the one above. It enables the solving of multiple equations i.e a system of equations:

${y_{1} = a_{11}x_{1} + \cdots +a_{1n}x_{n} + b_1}$ 

${y_{2} = a_{21}x_{1} + \cdots +a_{2n}x_{n} + b_2}$

${y_{3} = a_{31}x_{1} + \cdots +a_{3n}x_{n} + b_3}$

$\vdots$

$y_{n} = a_{n1}x_{1} + \cdots +a_{nn}x_{n} + b_n$

We stack up the $y_{i}$s into a list and call it the $y$ vector. And replace the coefficients with a matrix $A$:

$$\begin{bmatrix} a_{11} & \cdots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{n1} & \cdots & a_{nn} \end{bmatrix}$$

A system of equations is thus $$y = Ax + b$$. 

This is how linear algebra was first introduced to me. The topic goes into great depth into the properties of the matrix $A$ which captures all the information regarding a linear function. Solving the system refers to finding all the values in the vector $$x$$ that satisfy the equation.

The system of equations picture of $y=Ax$ is barely revelatory. It mechanically shows how entries of the vector $y$ depend on $x$. There are other interpretations of $A$. 

# Definition of linearity

## Linear map/tranformation/operation: Reinterpreting $$y = Ax$$

An operation/transformation/map $$\mathbb{L}(x)$$ takes in an input vector $$x$$ and outputs/transforms/maps to a vector $$y$$. 

Now's the time for the definition of linearity we've been building up to. A map/transformation/operation is linear if the superposition property holds:

$\mathbb{L}(\alpha x_1 + \beta x_2) = \alpha \mathbb{L}(x_1) + \beta \mathbb{L}(x_2)$

The equation tells us that if we take two vectors $$x_1$$ and $$x_2$$, scale them with $$\alpha$$ and $$\beta$$ and pass the result through the linear operator then the output is the same as if we had scaled each input and passed them through the operator separately and then added the two results to produce the output. This is a strong constraint; it means that the operator acts on each input independently.

In the system of equations picture of $$y = Ax$$, matrix A is a linear operator because it satisfies this superposition property. In fact, every linear function/transformation can be written as a matrix. 

Note that the earlier definition presented for a linear function (eg. ${f(x) = (-a_{2}/a_{1})x-b}$) does not satisfy the superposition property. It does not preserve the origin after the transformation. The earlier definition is technically "linear affine". The $$y = Ax$$ is the linear part; the $$b$$ term adds a translation like in the figure with the straight lines above. The term linear is often confused with affine (especially in calculus) so know that there is a difference. Henceforth, we'll use the definition for linear to mean "satisfying the superposition principle".

Before we start applying this definition of linearity we need to paint a better picture for what a vector is. 

# Generalizing vectors

## Abstract vector space: Where does a vector come from?

We just saw how the system of equations picture gives us a notion for what a vector is: a list of numbers. Geometrically, we can think of the list as points in a "space". If the numbers are real numbers and the list has $$n$$ elements, the vector belongs to a vector space of n-dimensional real-valued points ${\Re^{n}}$. 

Now the interesting bit is that a list of numbers is only one example for what a vector can be. You can define other objects as vectors belonging to some vector "space". We will see examples in a second but before that, here is the general definition of what a vector space is.

When you think of a vector space, you can think of it as a basket of four objects:
- A set $$V$$
- A definition for a sum operation $$+$$ that takes in two elements from $$V$$ and outputs another element inside $$V$$
- A definition for a scalar multiplication operation $$x$$ that takes in a real number and one elements from $$V$$ and outputs another element inside $$V$$
- A special "zero" object $$0$$ inside $$V$$

These four objects together need to satisfy eight properties/axioms which are listed [here](http://mathworld.wolfram.com/VectorSpace.html). 

One object that qualifes as a vector is a function $$f(a)$$. We can have a vector space that consists of two functions $$f(a)$$ and $$g(a)$$ and the distinguised element $$0$$ and define operations of addition ($$(f+g)(a)$$) and multiplication ($$(f*g)(a)$$) that satisfy the eight axioms. As a more specific example, we can define the vector space of polynomials of degree 2 as a set that contains the three functions $$f(x) = 1$$, $$g(x) = x$$ and $$h(x) = x^2$$. Differentiation $$d/dx$$ is an example of a linear operator that can act on this vector space of polynomials. We can re-write differentiation in a matrix form D as well! This brilliant 3blue1brown [video](https://www.youtube.com/watch?v=TgKwz5Ikpc8) captures how.

As an interesting aside, because random variables are functions (and hence vectors) too, ideas from linear algebra are easily applied to statistics as well. [Here's](https://www.randomservices.org/random/expect/Spaces.html) more info. 


### Basis of a vector space: Describing a vector space

The basis of a vector space is a set of vectors that can be thought of as building blocks of the entire vector space. The set of basis vectors is linearly independent -- none of the vectors inside the set can be written as a weighted sum (aka linear combination) of the other vectors.

For example, the standard basis for ${\Re^{3}}$ is a set of three vectors:
$$\begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix}$$, $$\begin{bmatrix}0 \\ 1 \\ 0\end{bmatrix}$$, $$\begin{bmatrix}0 \\ 0 \\ 1\end{bmatrix}$$. You can build any vector in ${\Re^{3}}$ using a linear combination of these three. 

The dimension of a vector space is defined by the number of basis vectors. The earlier example of a vector space of polynomials of degree 2 has 3 basis vectors ($$f(x), g(x), h(x)$$) and so it is 3 dimensional. A vector space of all polynomial functions is infinite dimensional. Note that a vector space can be have infinitely many basis sets.

### Inner product space: Equipping a vector space with a powerful operation

The dot product is a very useful linear operation that helps us intuit geometric notions of lengths and angles between vectors. It takes two vectors from a vector space and outputs a real number. The most common notation for the operation is $$\langle a, b \rangle = a \cdot b = a^T b = \sum_{i=1}^{n} a_{i} b_{i}$$. It is computed by summing up the element wise products of the two vectors.

Length of a vector $$a$$ is defined by the dot product of $$a$$ with itself $$|a| = \sqrt{\langle a, a \rangle}$$

The angle between two vectors $$a$$ and $$b$$ can also be defined by the dot product $$\langle a, b \rangle = |a| |b| cos(\theta)$$ or $$\theta = \cos^{-1} (\frac{\langle a, b \rangle}{|a| |b|})$$. 

Dot product also gives us the notion of orthogonality. When the dot product of two vectors is $$0$$, the two vectors are said to be orthogonal. i.e $$\langle a, b \rangle = |a| |b| cos(\theta) = 0$$, $$cos(\theta) = 0$$ or $$\theta = \pi/2 = 90^{\circ}$$. When the dot product is $|a| |b|$, the vectors are most "similar" or "co-directional" and when the vectors are orthogonal they are "perpendicular" or least "similar".

The dot product we have defined so far is a specific example of an inner product (for n-dimensional Euclidean spaces). Just as we did before in abstracting away vector spaces using axioms, we can do the same for inner products. For an operation to qualify as an inner product, it needs to satisfy three axioms listed [here](https://en.wikipedia.org/wiki/Inner_product_space#Definition). One of the axioms enforces the linearity constraint.

An inner product space is a vector space with a defined inner product operation. We generalized the definition of inner product so we can apply it to all kinds of vector spaces including vector spaces of functions.

A particulary convenient basis set is an orthonormal one. An orthonormal basis contains n orthogonal vectors (each vector orthogonal to every other vector) that have unit length (inner product of each basis vector with itself is $$1$$). For example, the standard basis defined above is orthonormal. Then we can write any vector $$a$$ as a weighted sum of the basis vectors $$v_{1}..v_{n}$$: $$a = w_{1}v_{1} + w_{2}v_{2} + ... + w_{n}v_{n}$$. The next natural question then is how to we find the weights $$w_{i}$$? If $$a$$ is expressed in one basis set eg. standard basis set, then to find the weights in the new basis set of $$v_{i}s$$, we simply compute the inner product between $$a$$ and $$v_{i}$$. $$w_{i} = \langle v_i, a \rangle$$. These new $$w_{i}$$s are the elements of the same vector expressed in the $$v_{i}$$ basis. We have effectively performed a *change of basis.*


With all the generalizations and abstractions defined, we are ready to apply our general linearity definition and general inner product spaces to understanding the Fourier transform.

## Example: Fourier transforms as linear transforms

Fourier transforms are ubiquitous in engineering. They allow us to break down any signal/function into a summation of infinitely many sinusoid functions of varying amplitudes and phases. This animation shows how a square waveform is broken into a sum of sines and cosines: {{< youtube r4c9ojz6hJg >}}

The equation for the transformation looks like:
 
$\mathscr{F}[\{f(t)\]}=\hat{f}(\omega)$

$\mathscr{F}[\{f(t)\]}=\int_{-\infty}^{\infty} f(t) e^{-i \omega t} dt$


This is a scary equation at first but with the ideas of vector spaces and linearity, we can interpret this quickly. 

- It maps a function (aka vector) to another function. The input function is often interpreted as a function of time $$t$$ and the output as a function of frequency $$\omega$$. The input function $$f(t)$$ comes from the vector space of functions $$f(t)$$ which map a real number (time) to a complex number (eg. signal value, can also just be real). Similarly, the output function $$\hat{f}(\omega)$$ belongs to the vector space of complex valued functions. $$\hat{f}(\omega)$$ takes in a real valued frequency and outputs a complex value.

- The complex exponential function i.e $$e^{i\omega t}$$ is very handy in simplifying a lot of math involving sines and cosines. First, using Euler's identity, the complex exponential is simply $$cos(\omega t) + i sin (\omega t)$$. We can write a cosine or a sine in terms of the complex exponential. $$cos(\theta) = \frac{e^{i\theta}+e^{-i\theta}}{2}$$ and $$sin(\theta) = \frac{e^{i\theta}-e^{-i\theta}}{2i}$$.
One example of how it simplifies math is when adding a phase to a cosine wave, $$e^{i (\theta + \phi)} = e^{i\theta}e^{i\phi}$$. If we tried doing it using trigonomtry, we could get the following complicated relationship: $$cos(\theta + \phi) = cos(\theta)cos(\phi)-sin(\theta)sin(\phi)$$. 

- The notion of frequency comes from the fact that the argument for the complex exponential can be interpreted as the rate at which the sine/cosine wave making up the complex exponential completes a full cycle i.e if the sine wave takes T seconds to complete a period $2\pi$, it's frequency is $\omega = 2\pi/T$.

- Aside from practical considerations, the choice of complex expontial might be understood better with the notion of eigenfunctions. Eigenvectors/eigenfunctions are vector objects that pass through a linear operator with only a scaling factor (called the eigenvalue). The set of eigenvectors is a complete basis for the vector space -- we can write every vector as a linear combination of eigenvectors. If we know how a linear operator acts on each eigenvector, then we know everything about the operator; we can decompose any input signal into the basis of eigenvectors, apply the appropriate scaling factors of the linear operator and re-combine to produce the output signal. Complex exponentials are eigenfunctions of linear time-invariant operators. See [here](https://ptolemy.berkeley.edu/eecs20/week9/lti.html).

- The integral of the product of the two functions should look similar to the dot product of the two vectors we saw earlier. In fact, the integral is a valid inner product on the vector space of functions. For a finite dimension list of numbers picture of two vectors (i.e vectors in $$\Re^{n}$$), we defined the inner product as $$\langle a, b \rangle = a \cdot b = \sum_{i=1}^{n} a_{i}  b_{i}$$. The analogue for an infinite dimensional function space uses calculus: $\langle f, g \rangle = f \cdot g = f(a)g(a)dx + f(a+dx)g(a+dx)dx + f(a+2dx)g(a+2dx) ... = \int_{a}^{b} f(x)g(x)dx$. What we have here with Fourier transforms though are complex-valued functions and to meet the axioms for inner products, we need to modify the inner product to be:
$\langle f, g \rangle = \int_{a}^{b} f^{*}(x)g(x)dx$. In words, the inner product of two complex-valued functions is given by the integral over the product of the complex-conjugate of one function with the other function.

- The Fourier transform is thus the inner product between two functions -- the input function and the complex exponential evaluated at a particular frequency value. $$\hat{f}(\omega) = \langle e^{i \omega t}, f \rangle = \int_{-\infty}^{\infty} f(t)e^{-i \omega t}dt$$. 

- To summarize, the Fourier transform is a *linear* operator. Specifically it is an inner product linear operator. It changes the basis of our function/signal $$f(t)$$ into the basis of the complex exponential functions $$e^{i \omega t}$$. It tells you for a particular given frequency $$\omega$$, how similar is the input function to a complex exponential of frequency $$\omega$$. The output of the transform $\hat{f}(\omega)$ then reveals the coefficients of the different complex exponentials evaluated at different $\omega$s. These coefficients are complex-valued and give us a frequency spectrum picture for any signal.

## Conclusion

We started off by immediately replacing the straight line picture of linearity with that of hyperplanes. With the system of equations $$y = Ax$$, we saw concrete examples of what vectors are (list of numbers or points in space) and what a linear function as a matrix $$A$$ is (coefficients of linear equations relating input and output vectors). 

We then defined a linear operator (eg. $$A$$ in $$y=Ax$$) using the idea of linear superposition-- the sum of the scaled outputs of a linear operator to different outputs is equivalent to the output of the linear operator to the sum of the scaled input signals.

The big jump in abstraction comes with the definition of vector spaces and how vectors can be more than just lists of numbers. Specifically, we saw how functions can be n-dimensional vectors coming from some vector space of functions. A vector space can be described by a set of basis vectors the number of which define the dimensionality of the vector space. 

Last, we applied our generalization of vector spaces and linearity to gain a new perspective of the popular Fourier transform; as an inner product between the input function and a complex exponential.