import numpy as np

def set_pair_potentials_params(job, morse, table, table_func, types):
    # set Morse potential parameters
    d_colors = job.sp.d_colors
    morse_d0s = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., d_colors[0], 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., d_colors[1], 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., d_colors[2]],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., d_colors[0], 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., d_colors[1], 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., d_colors[2], 0., 0., 0., 0.]])
    
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                morse.params[(type_i,type_j)] = dict(alpha=job.sp.alpha, D0=job.sp.D0*morse_d0s[i,j], r0=job.sp.r0)

                         #['A', 'B', 'AM', 'AP1', 'AP2', 'AP3', 'BM', 'BP1', 'BP2', 'BP3']
    morse_r_cuts = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                if morse_r_cuts[i,j] == 0.0:
                    morse.r_cut[(type_i,type_j)] = 0


    # Set lj parameters
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                table.params[(type_i,type_j)] = dict(r_min = 0, U=table_func[0], F=table_func[1])

                         #['A', 'B', 'AM', 'AP1', 'AP2', 'AP3', 'BM', 'BP1', 'BP2', 'BP3']
    table_r_cuts = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                if table_r_cuts[i,j] == 0.0:
                    table.params[(type_i,type_j)] = dict(r_min=0, U=[0], F=[0])
                    table.r_cut[(type_i,type_j)] = 0