# Van-der-Pol-Solver
A solver for the Van der Pol equation with methods of creating an animation.  Based on code by ishidur (https://github.com/ishidur/Van_der_Pol_visualizer)

The Van der Pol equation is a nonlinear second-order dynamical system with a unique, stable limit cycle.  The equation is commonly seen in one of the following three forms:

1. <img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7Bd%5E2x%7D%7Bdt%5E2%7D%20%20%2B%20%28x%5E2%20-%201%29%20%5Cfrac%7Bdx%7D%7Bdt%7D%20%20%2B%20x%20%3D%200&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \frac{d^2x}{dt^2}  + (x^2 - 1) \frac{dx}{dt}  + x = 0" width="208" height="46" />


2. <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Bcases%7D%20%5Cfrac%7Bdx%7D%7Bdt%7D%20%3D%20y%20%20%26%20%5C%5C%20%5Cfrac%7Bdy%7D%7Bdt%7D%20%3D%20-x%20%2B%20%281%20-%20x%5E2%29y%20%26%20%5Cend%7Bcases%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\begin{cases} \frac{dx}{dt} = y  & \\ \frac{dy}{dt} = -x + (1 - x^2)y & \end{cases} " width="178" height="53" />


3. <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Bcases%7D%20%5Cfrac%7Bdx%7D%7Bdt%7D%20%3D%20z%20-%20%20%5Cbigg%28%5Cfrac%7Bx%5E3%7D%7B3%7D%20-%20x%5Cbigg%29%20%20%20%26%20%5C%5C%20%5Cfrac%7Bdz%7D%7Bdt%7D%20%3D%20-x%20%26%20%5Cend%7Bcases%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\begin{cases} \frac{dx}{dt} = z -  \bigg(\frac{x^3}{3} - x\bigg)   & \\ \frac{dz}{dt} = -x & \end{cases} " width="162" height="81" />



-- VDP_y_funcs.py: Functions used to solve, plot, and animate the solutions to the Van der Pol equation in the form of Eqn. 2

-- VDP_funcs.py: Functions used to solve, plot, and animate the solutions to the Van der Pol equation in the form of Eqn. 3

--- Both may be used to solve the Van der Pol equation in the form of Eqn. 1

-- VDP_z_vs_x.ipynb/VDP_z_vs_y.ipynb: Demonstrates the creation of a phase plot for the z and y equations, respectively.
--- These clearly show the existence and stability of the limit cycle

-- VDP_z_vs_x_Animation.ipynb/VDP_z_vs_y_Animation.ipynb: Demonstrates the creation of a phase plot animation for the z and y equations, respectively.
WARNING: THESE ARE EXTREMELY SLOW!  I will be updating this repository with videos of the animations as soon as they finish rendering

-- Several static plots for various initial conditions are shown in the 'figures' directory

-- Several animations for various initial conditions are shown in the 'figures' directory
