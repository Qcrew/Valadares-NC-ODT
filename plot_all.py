import numpy as np

from fig2.plot_fig2b import plot_fig2b
from fig2.plot_fig2c import plot_fig2c
from fig3.plot_fig3c_wigner import plot_fig3c_wigner
from fig3.plot_fig3c_char import plot_fig3c_char
from fig4.plot_fig4a_high_kerr import plot_fig4a_high_kerr
from fig4.plot_fig4a_low_kerr import plot_fig4a_low_kerr
from fig4.plot_fig4b_high_chi import plot_fig4b_high_chi
from fig4.plot_fig4b_low_chi import plot_fig4b_low_chi
from supp_fig2.plot_supp_fig2 import plot_supp_fig2


MAIN_DIR = "/Users/kyle/Documents/qcrew/on-demand-transposition-across-light-matter-interaction-regimes-in-bosonic-cQED"
SAVE_DATA = False

fig2b_data = plot_fig2b(MAIN_DIR)
fig2c_data = plot_fig2c(MAIN_DIR)
fig3c_char_data = plot_fig3c_char(MAIN_DIR)
fig3c_wigner_data = plot_fig3c_wigner(MAIN_DIR)
fig4a_low_kerr_data = plot_fig4a_low_kerr(MAIN_DIR)
fig4a_high_kerr_data = plot_fig4a_high_kerr(MAIN_DIR)
fig4b_low_chi_data = plot_fig4b_low_chi(MAIN_DIR)
fig4b_high_chi_data = plot_fig4b_high_chi(MAIN_DIR)
supp_fig2_data = plot_supp_fig2(MAIN_DIR)

if SAVE_DATA:
    np.savez(
        "Source Data.npz",
        fig2b=fig2b_data,
        fig2c=fig2c_data,
        fig3c_char=fig3c_char_data,
        fig3c_wigner=fig3c_wigner_data,
        fig4a_low_kerr=fig4a_low_kerr_data,
        fig4a_high_kerr=fig4a_high_kerr_data,
        fig4b_low_chi=fig4b_low_chi_data,
        fig4b_high_chi=fig4b_low_chi_data,
        supp_fig2=supp_fig2_data,
    )
