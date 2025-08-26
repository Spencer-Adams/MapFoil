import sympy as sp

# Define variables
print("Defining symbols...")
theta = sp.symbols('theta', real=True)
R, epsilon, zeta0, alpha, Gamma, Vinf = sp.symbols('R epsilon zeta0 alpha Gamma Vinf', real=True)
i = sp.I

# Define expressions
print("Building expressions...")
zeta = R * sp.exp(i * theta)
a = Vinf * (sp.exp(-i * alpha) + i * Gamma / (2 * sp.pi * Vinf) * (1 / zeta) - R**2 * sp.exp(i * alpha) / zeta**2)
b = 1 - (R - epsilon)**2 / (zeta + zeta0)**2
c = Vinf * (2 * R * sp.exp(i * alpha) - i * Gamma * sp.exp(i * theta) / (2 * sp.pi * Vinf)) / (R**2 * sp.exp(2 * i * theta))
d = 2 * sp.exp(i * theta) * (R - epsilon)**2 / (zeta + zeta0)**3

# Conjugates
print("Calculating conjugates...")
a_conj = sp.conjugate(a)
b_conj = sp.conjugate(b)

# Build integrand
print("Building integrand...")
numerator = b_conj**2 * a_conj**2
denominator = b**4 * b_conj
multiplier = b**2 * a * c - a**2 * b * d
integrand = numerator / denominator * multiplier

# Define integration bounds
theta0 = sp.symbols('theta0')  # This is your aft-stagnation theta

print("Starting symbolic integration. This may take a while...")
S = -R/16 * sp.integrate(integrand, (theta, theta0, 2*sp.pi + theta0))
print("Integration complete.")
print("Result S: \n", S)
