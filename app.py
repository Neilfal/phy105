import streamlit as st
import expectation_values_normalized as ev

LIMITS = {1: (0.0, 1.0), 2: (0.0, 1.0), 3: (0.0, 1.0),
          4: (-6.0, 6.0), 5: (-6.0, 6.0), 6: (-6.0, 6.0),
          7: (-15.0, 15.0), 8: (-8.0, 8.0),
          9: (0.0, 30.0), 10: (0.0, 60.0)}

st.set_page_config(page_title="Expectation Values", page_icon="⚛️")
st.title("Expectation Values in Quantum Mechanics")
st.caption("⟨x⟩, ⟨p⟩, ⟨p²⟩ and ⟨E⟩ for 10 predefined wavefunctions (ħ = m = 1). "
           "For the hydrogen states, x means the radius r.")

choice = st.selectbox("Wavefunction", list(range(1, 11)),
                      format_func=lambda i: f"{i}. {ev.NAMES[i - 1].split('[')[0].strip()}")

lo, hi = LIMITS[choice]
col1, col2 = st.columns(2)
a = col1.number_input("Lower limit", value=lo, key=f"a{choice}", format="%.4f")
b = col2.number_input("Upper limit", value=hi, key=f"b{choice}", format="%.4f")

if st.button("Calculate", type="primary"):
    if a >= b:
        st.error("Lower limit must be smaller than upper limit.")
    else:
        norm = ev.normalization(choice, a, b)
        if norm <= 1e-12:
            st.error("The normalization integral is zero or too small. "
                     "Choose limits that contain some probability density.")
        else:
            # Each function returns the numerator of <A>; divide by norm.
            x_avg = ev.expectation_position(choice, a, b) / norm
            p_avg = ev.expectation_momentum(choice, a, b) / norm
            p2_avg = ev.expectation_p_squared(choice, a, b) / norm
            e_avg = ev.expectation_energy(choice, a, b) / norm

            full = abs(norm - 1.0) <= 1e-3
            tx, tp, tp2, te = ev.exact_values(choice) if full else (None,) * 4
            fmt = lambda v: "–" if v is None else f"{v:.4f}"
            r4 = lambda z: f"{round(z.real, 4) + 0.0:.4f}"

            st.write(f"Normalization integral ∫|ψ|² dx = **{norm:.6f}**")
            st.table({
                "Quantity": ["⟨x⟩", "⟨p⟩", "⟨p²⟩", "⟨E⟩"],
                "Numerical": [r4(x_avg), ev.format_complex(p_avg), r4(p2_avg), r4(e_avg)],
                "Exact (full domain)": [fmt(tx), fmt(tp), fmt(tp2), fmt(te)],
            })
            if not full:
                st.info("Values are divided by the norm, so they are normalized over the "
                        "selected interval [a, b]. They may differ from full-domain values "
                        "because the limits may exclude part of the quantum state.")
