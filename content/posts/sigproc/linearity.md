+++ 
date = "2018-11-05"
title = "What even is linear?"
markup = "mmark"
+++

I am currently taking a class on linear dynamical systems. The three big questions I really wanted an in-depth answer to are: what is linear? what is dynamics? What is a system? Today, I want to tackle the first question on linearity. There are so many ideas revolving around linearity: linear equations, linear functions, linear systems, linear algebra, linear regression, linear filter etc. We all think straight line when we think of the word linear. My first strong association of the word linear is from physics -- the rectilinear propagation of light: that light travels in straight line. But what do straight lines have to do with all those listed ideas? Can we define linearity in a context independent manner? In this post, we’ll start from elementary math and build up a definition that can be applied whenever you encounter the word linear. My goal in doing this exercise was to replace the straight line mental picture with some other picture that makes more sense, and is more useful.

## Linear equations

An equation relates variables with equality and takes the following form for n variables:

${\displaystyle a_{1}x_{1}+\cdots +a_{n}x_{n}+b=0}$


- The highest degree in this equation is 1. 

- Solving this equation refers to finding values for the variables that make the equation hold true. The solution set contains values of variables that When n = 2, the solution to the equation lives on a straight line with n = 2. To see this geometrically, we can plot the pair of variables in a two dimensional graph.

![linear_equation](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Linear_Function_Graph.svg/800px-Linear_Function_Graph.svg.png)

- The generalization of the solution set is a hyperplane (line in 2D plane, a plane in 3D, a cube in 4D etc.). So linear does not always imply straight lines! But it's helpful to reason about and visualize multiple variables by reducing them to the case of n=2 and investigate behavior in terms of straight lines.

Linear function

A function is an association of an element x in one set X to another element in set Y. It is most commonly specified as an equation:

$y = f(x)$

If f(x) takes us to the form of (1) then, it’s a linear function. We can also generalize a linear function as having n independent variables:

$Y = f(x_1, x_2...x_n)$

We can stack up $x_1...x_n$ into a list of numbers and call the new list object a vector. The vector is said to be n dimensional. 
