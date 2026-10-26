"""
Calculation of Expectation Values in Quantum Mechanics
------------------------------------------------------
Group III - Problem 2

The user chooses one of 10 PREDEFINED wavefunctions, then enters the
limits of integration. The program finds <x>, <p>, <p^2> and <E> using

    <A> = integral of  psi*(x) [ A_operator psi(x) ] dx

    x operator   : x
    p operator   : -i*hbar * d/dx
    p^2 operator : -hbar^2 * d^2/dx^2
    H operator   : -(hbar^2 / 2m) * d^2/dx^2 + V(x)

Units: hbar = 1, m = 1 (natural units).
(For the hydrogen states the variable x means the radius r.)
"""

import math
import cmath

hbar = 1.0
m = 1.0
N_STRIPS = 4000      # number of strips for the trapezoidal rule
H_STEP = 1e-5        # step size for the derivatives


# ---------------------------------------------------------------
# 1. The 10 predefined wavefunctions
# ---------------------------------------------------------------
def box_state(x, n):
    """Particle in a box of length 1: sqrt(2) sin(n pi x), zero outside."""
    if x < 0 or x > 1:
        return 0.0
    return math.sqrt(2.0) * math.sin(n * math.pi * x)


def psi_1(x):
    """Infinite well, n = 1 (L = 1)"""
    return box_state(x, 1)


def psi_2(x):
    """Infinite well, n = 2 (L = 1)"""
    return box_state(x, 2)


def psi_3(x):
    """Infinite well, n = 3 (L = 1)"""
    return box_state(x, 3)


def psi_4(x):
    """Harmonic oscillator, n = 0 (m = omega = 1)"""
    return (1 / math.pi) ** 0.25 * math.exp(-x * x / 2)


def psi_5(x):
    """Harmonic oscillator, n = 1"""
    return math.sqrt(2.0) * (1 / math.pi) ** 0.25 * x * math.exp(-x * x / 2)


def psi_6(x):
    """Harmonic oscillator, n = 2"""
    return (2 * x * x - 1) / (math.sqrt(2.0) * math.pi ** 0.25) * math.exp(-x * x / 2)


def psi_7(x):
    """Free Gaussian wave packet, width sigma = 2, k0 = 0"""
    sigma = 2.0
    return (1 / (math.pi * sigma ** 2)) ** 0.25 * math.exp(-x * x / (2 * sigma ** 2))


def psi_8(x):
    """Free Gaussian wave packet with momentum: sigma = 1, k0 = 2 (complex)"""
    return (1 / math.pi) ** 0.25 * math.exp(-x * x / 2) * cmath.exp(2j * x)


def psi_9(r):
    """Hydrogen 1s radial function u(r) = 2 r exp(-r)"""
    if r < 0:
        return 0.0
    return 2 * r * math.exp(-r)


def psi_10(r):
    """Hydrogen 2s radial function u(r) = r (2 - r) exp(-r/2) / (2 sqrt 2)"""
    if r < 0:
        return 0.0
    return r * (2 - r) * math.exp(-r / 2) / (2 * math.sqrt(2.0))


def wavefunction(choice, x):
    """Return psi(x) for the chosen wavefunction number."""
    if choice == 1:
        return psi_1(x)
    if choice == 2:
        return psi_2(x)
    if choice == 3:
        return psi_3(x)
    if choice == 4:
        return psi_4(x)
    if choice == 5:
        return psi_5(x)
    if choice == 6:
        return psi_6(x)
    if choice == 7:
        return psi_7(x)
    if choice == 8:
        return psi_8(x)
    if choice == 9:
        return psi_9(x)
    return psi_10(x)


def potential(choice, x):
    """Potential energy V(x) that goes with each wavefunction."""
    if choice in (4, 5, 6):          # harmonic oscillator
        return 0.5 * m * x * x
    if choice in (9, 10):            # hydrogen (Coulomb)
        if x <= 0:
            return 0.0
        return -1.0 / x
    return 0.0                       # box (inside) and free packets


# ---------------------------------------------------------------
# 2. Menu text, suggested limits and exact values
# ---------------------------------------------------------------
NAMES = [
    "Infinite well, n = 1 (L = 1)       [limits 0 to 1]",
    "Infinite well, n = 2 (L = 1)       [limits 0 to 1]",
    "Infinite well, n = 3 (L = 1)       [limits 0 to 1]",
    "Harmonic oscillator, n = 0         [limits -6 to 6]",
    "Harmonic oscillator, n = 1         [limits -6 to 6]",
    "Harmonic oscillator, n = 2         [limits -6 to 6]",
    "Gaussian packet, sigma=2, k0=0     [limits -15 to 15]",
    "Gaussian packet, sigma=1, k0=2     [limits -8 to 8]",
    "Hydrogen 1s radial function        [limits 0 to 30]",
    "Hydrogen 2s radial function        [limits 0 to 60]",
]


def exact_values(choice):
    """Exact <x>, <p>, <p^2>, <E> when the limits cover the whole state."""
    pi2 = math.pi ** 2
    if choice == 1:
        return 0.5, 0.0, pi2, pi2 / 2
    if choice == 2:
        return 0.5, 0.0, 4 * pi2, 2 * pi2
    if choice == 3:
        return 0.5, 0.0, 9 * pi2, 4.5 * pi2
    if choice == 4:
        return 0.0, 0.0, 0.5, 0.5
    if choice == 5:
        return 0.0, 0.0, 1.5, 1.5
    if choice == 6:
        return 0.0, 0.0, 2.5, 2.5
    if choice == 7:
        return 0.0, 0.0, 0.125, 0.0625
    if choice == 8:
        return 0.0, 2.0, 4.5, 2.25
    if choice == 9:
        return 1.5, 0.0, 1.0, -0.5
    return 6.0, 0.0, 0.25, -0.125


# ---------------------------------------------------------------
# 3. Derivatives (central difference formulas)
# ---------------------------------------------------------------
def first_derivative(choice, x):
    """d(psi)/dx"""
    h = H_STEP
    return (wavefunction(choice, x + h) - wavefunction(choice, x - h)) / (2 * h)


def second_derivative(choice, x):
    """d^2(psi)/dx^2"""
    h = H_STEP
    return (wavefunction(choice, x + h)
            - 2 * wavefunction(choice, x)
            + wavefunction(choice, x - h)) / (h * h)


# ---------------------------------------------------------------
# 4. Numerical integration (trapezoidal rule)
# ---------------------------------------------------------------
def integrate(values, dx):
    total = 0.0
    for i in range(len(values) - 1):
        total += (values[i] + values[i + 1]) / 2.0 * dx
    return total


# ---------------------------------------------------------------
# 5. Normalisation and expectation values
#    (psi may be complex, so psi*(x) = psi.conjugate())
# ---------------------------------------------------------------
def normalization(choice, a, b):
    dx = (b - a) / N_STRIPS
    values = []
    for i in range(N_STRIPS + 1):
        x = a + i * dx
        psi = wavefunction(choice, x)
        values.append(psi.conjugate() * psi)
    return integrate(values, dx).real


def expectation_position(choice, a, b):
    """<x> = integral psi* x psi dx"""
    dx = (b - a) / N_STRIPS
    values = []
    for i in range(N_STRIPS + 1):
        x = a + i * dx
        psi = wavefunction(choice, x)
        values.append(psi.conjugate() * x * psi)
    return integrate(values, dx)


def expectation_momentum(choice, a, b):
    """<p> = integral psi* (-i hbar d/dx) psi dx"""
    dx = (b - a) / N_STRIPS
    values = []
    for i in range(N_STRIPS + 1):
        x = a + i * dx
        psi = wavefunction(choice, x)
        values.append(psi.conjugate() * first_derivative(choice, x))
    return -1j * hbar * integrate(values, dx)


def expectation_p_squared(choice, a, b):
    """<p^2> = integral psi* (-hbar^2 d^2/dx^2) psi dx"""
    dx = (b - a) / N_STRIPS
    values = []
    for i in range(N_STRIPS + 1):
        x = a + i * dx
        psi = wavefunction(choice, x)
        values.append(psi.conjugate() * (-hbar ** 2) * second_derivative(choice, x))
    return integrate(values, dx)


def expectation_energy(choice, a, b):
    """<E> = integral psi* [ -(hbar^2/2m) d^2/dx^2 + V ] psi dx"""
    dx = (b - a) / N_STRIPS
    values = []
    for i in range(N_STRIPS + 1):
        x = a + i * dx
        psi = wavefunction(choice, x)
        h_psi = (-(hbar ** 2) / (2 * m) * second_derivative(choice, x)
                 + potential(choice, x) * psi)
        values.append(psi.conjugate() * h_psi)
    return integrate(values, dx)


# ---------------------------------------------------------------
# 6. Input helpers
# ---------------------------------------------------------------
def get_float(message):
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("  Please enter a number.")


def get_choice():
    while True:
        try:
            value = int(input("Enter your choice (1-10): "))
            if 1 <= value <= 10:
                return value
            print("  Choose a number from 1 to 10.")
        except ValueError:
            print("  Please enter a whole number.")


def format_complex(z):
    real = round(z.real, 4) + 0.0      # + 0.0 turns -0.0 into 0.0
    imag = round(z.imag, 4) + 0.0
    return f"{real} + ({imag})i"


# ---------------------------------------------------------------
# 7. Main program
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Expectation values  (hbar = m = 1)")
    print("=" * 60)

    while True:
        print("\nPredefined wavefunctions:")
        for i in range(10):
            print(f"  {i + 1:2d}. {NAMES[i]}")
        print()

        choice = get_choice()
        a = get_float("Enter lower limit of integration: ")
        b = get_float("Enter upper limit of integration: ")
        while a >= b:
            print("  Lower limit must be smaller than upper limit.")
            a = get_float("Enter lower limit of integration: ")
            b = get_float("Enter upper limit of integration: ")

        norm = normalization(choice, a, b)

        # A zero or extremely small norm cannot be used as a denominator.
        if norm <= 1e-12:
            print("\nThe normalization integral is zero or too small.")
            print("Choose limits that contain some probability density.")
            again = input("\nCalculate again? (y/n): ").strip().lower()
            if again != "y":
                print("Program finished.")
                break
            continue

        # Each expectation function returns the numerator of <A>.
        # Divide by norm to get the normalized expectation over [a, b].
        x_avg = expectation_position(choice, a, b) / norm
        p_avg = expectation_momentum(choice, a, b) / norm
        p2_avg = expectation_p_squared(choice, a, b) / norm
        e_avg = expectation_energy(choice, a, b) / norm

        print()
        print(f"Normalization integral |psi|^2 dx = {norm:.6f}")
        print("Normalized expectation values over the selected interval:")
        print(f"<x>   = {round(x_avg.real, 4) + 0.0:.4f}")
        print(f"<p>   = {format_complex(p_avg)}")
        print(f"<p^2> = {round(p2_avg.real, 4) + 0.0:.4f}")
        print(f"<E>   = {round(e_avg.real, 4) + 0.0:.4f}")

        if abs(norm - 1.0) > 1e-3:
            print("\nNote: the expectation values above are divided by the norm, so")
            print("they are normalized over the selected interval [a, b].")
            print("They may differ from full-domain values because the selected")
            print("limits may exclude part of the quantum state.")
        else:
            tx, tp, tp2, te = exact_values(choice)
            print("\nExact full-domain values for comparison:")
            print(f"<x> = {tx:.4f}   <p> = {tp:.4f}   <p^2> = {tp2:.4f}   <E> = {te:.4f}")

        again = input("\nCalculate again? (y/n): ").strip().lower()
        if again != "y":
            print("Program finished.")
            break


if __name__ == "__main__":
    main()
