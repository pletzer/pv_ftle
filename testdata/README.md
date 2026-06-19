Test files to generate PALM-FTLE analysis in "Mimetic interpolation for finite-time Lyapunov exponent
computation near irregular domain boundaries"

test.nc: ncks -d time,2210,2220 -v u_xy,v_xy,w_xy ~/mnt/u/blf_night_loc1_4m_xy_N04.002.nc test.nc

test_ftle_2220.vtr: palm_ftle test.nc --time-index 10 --vtkout test_ftle_2220.vtr
