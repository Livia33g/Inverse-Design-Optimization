import json
import sys


def save_polymer_distribution():
    
    import signac
    from statistics import mean, stdev

    project = signac.get_project()
    polymer_size_by_kT = {}
    concentrations = set({})
    for job in project:
        concentrations.add(job.sp.concentration)

    for concentration in concentrations:
        polymer_size_by_kT[f"concentration={concentration}"] = {}
        polymer_sizes = set({})
        max_polymer_size = 0
        for job in project.find_jobs({'concentration': concentration}):
            with job:
                with open("polymer_dist.json", "r") as json_file:
                    data = json.load(json_file)
                    if f"kT={job.sp.kT}" in polymer_size_by_kT[f"concentration={concentration}"].keys():
                        polymer_size_by_kT[f"concentration={concentration}"][f"kT={job.sp.kT}"][f"replica {job.sp.replica}"] = data["Averages"]
                    else:
                        polymer_size_by_kT[f"concentration={concentration}"][f"kT={job.sp.kT}"] = {f"replica {job.sp.replica}": data["Averages"]}
                    if len(data["Averages"]) > max_polymer_size:
                        max_polymer_size += 1
                        for polymer_size in data["Averages"]:
                            polymer_sizes.add(polymer_size)
        
        for kT in polymer_size_by_kT[f"concentration={concentration}"]:
            Data = {polymer_size: [] for polymer_size in polymer_sizes}
            for replica in polymer_size_by_kT[f"concentration={concentration}"][kT]:
                for polymer in Data.keys():
                    if polymer in polymer_size_by_kT[f"concentration={concentration}"][kT][replica].keys():
                        Data[polymer].append(polymer_size_by_kT[f"concentration={concentration}"][kT][replica][polymer])
                    else: 
                        Data[polymer].append(0)

            Average = {}
            Stdev = {}
            for polymer in Data:
                Average[polymer] = round(mean(Data[polymer]))
                if len(Data[polymer]) > 1:
                    Stdev[polymer] = stdev(Data[polymer])
                else:
                    Stdev[polymer] = 0
            polymer_size_by_kT[f"concentration={concentration}"][kT]["Average"] = Average
            polymer_size_by_kT[f"concentration={concentration}"][kT]["Std_Dev"] = Stdev

    print(polymer_size_by_kT)

    with open("polymer_dist_by_kT.json", "w") as json_file:
        json.dump(polymer_size_by_kT, json_file, indent=4, sort_keys=True)



def plot_polymer_distribution():
    
    import matplotlib.pyplot as plt
    import numpy as np

    with open("polymer_dist_by_kT.json", "r") as json_file:
        distribution = json.load(json_file)

    for concentration in distribution.keys():
        print(concentration)
        kT = []
        monomer_numbers = {}
        monomer_number_errors = {}

        for key in distribution[concentration]:
            kT.append(float(key[3:]))
            all_monomers = sum(float(distribution[concentration][key]["Average"][num_monomers]) for num_monomers in distribution[concentration][key]["Average"].keys())

            for num_monomers in  distribution[concentration][key]["Average"].keys():
                if num_monomers not in monomer_numbers.keys():
                    monomer_numbers[num_monomers] = []
                monomer_numbers[num_monomers].append(float(distribution[concentration][key]["Average"][num_monomers])/all_monomers)
            
            for num_monomers in  distribution[concentration][key]["Std_Dev"].keys():
                if num_monomers not in monomer_number_errors.keys():
                    monomer_number_errors[num_monomers] = []
                monomer_number_errors[num_monomers].append(float(distribution[concentration][key]["Std_Dev"][num_monomers])/all_monomers)

        # Create plots
        fig, ax = plt.subplots()

        # Plot data
        dimer_yield = None
        for i, num_monomers in enumerate(monomer_numbers.keys()):
            if i <= 1: # This only plots yield of one and two monomer units
                ax.errorbar(1/np.array(kT)[::-1], np.array(monomer_numbers[num_monomers])[::-1], yerr=np.array(monomer_number_errors[num_monomers])[::-1], capsize=2.0, label=f"{num_monomers.split(" ")[0]} monomers")
                if i == 1:
                    dimer_yields = np.array(monomer_numbers[num_monomers])[::-1]
                    dimer_errors = np.array(monomer_number_errors[num_monomers])[::-1]
        plt.axvline(x=1.0, ls="--")
        plt.text(1/np.array(kT)[::-1][0]+0.02, dimer_yields[0], f'{dimer_yields[0]} \n $\pm$ {dimer_errors[0]}', bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))

        # Axes Labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("1/kT")
        ax.set_ylabel("Fraction of Polymers")
        ax.set_title(f"Polymer Size Distributions by kT for {concentration}")
        ax.legend()
        plt.savefig(f"polymer_distribution_{concentration}.svg")
        plt.close()


def save_and_plot_full_traj():
    import signac
    from statistics import mean, stdev

    project = signac.get_project()
    polymer_size_by_kT = {}
    concentrations = set({})
    for job in project:
        concentrations.add(job.sp.concentration)

    for concentration in concentrations:
        polymer_size_by_kT[f"concentration={concentration}"] = {}
        polymer_sizes = set({})
        max_polymer_size = 0
        for job in project.find_jobs({'concentration': concentration}):
            with job:
                with open("polymer_dist_full_traj.json", "r") as json_file:
                    data = json.load(json_file)
                    for frame in data.keys():
                        if frame in polymer_size_by_kT[f"concentration={concentration}"].keys():
                            if f"kT={job.sp.kT}" in polymer_size_by_kT[f"concentration={concentration}"][frame].keys():
                                polymer_size_by_kT[f"concentration={concentration}"][frame][f"kT={job.sp.kT}"][f"replica {job.sp.replica}"] = data[frame]
                            else:
                                polymer_size_by_kT[f"concentration={concentration}"][frame][f"kT={job.sp.kT}"] = {f"replica {job.sp.replica}": data[frame]}
                        else:
                            polymer_size_by_kT[f"concentration={concentration}"][frame] = {f"kT={job.sp.kT}": {f"replica {job.sp.replica}": data[frame]}}
                        if len(data[frame]) > max_polymer_size:
                            max_polymer_size += 1
                            for polymer_size in data[frame]:
                                polymer_sizes.add(polymer_size)
        for frame in polymer_size_by_kT[f"concentration={concentration}"]:
            for kT in polymer_size_by_kT[f"concentration={concentration}"][frame]:
                Data = {polymer_size: [] for polymer_size in polymer_sizes}
                for replica in polymer_size_by_kT[f"concentration={concentration}"][frame][kT]:
                    for polymer in Data.keys():
                        if polymer in polymer_size_by_kT[f"concentration={concentration}"][frame][kT][replica].keys():
                            Data[polymer].append(polymer_size_by_kT[f"concentration={concentration}"][frame][kT][replica][polymer])
                        else: 
                            Data[polymer].append(0)

                Average = {}
                Stdev = {}
                for polymer in Data:
                    Average[polymer] = round(mean(Data[polymer]))
                    if len(Data[polymer]) > 1:
                        Stdev[polymer] = stdev(Data[polymer])
                    else:
                        Stdev[polymer] = 0
                polymer_size_by_kT[f"concentration={concentration}"][frame][kT]["Average"] = Average
                polymer_size_by_kT[f"concentration={concentration}"][frame][kT]["Std_Dev"] = Stdev

                # Sort kT
                polymer_size_by_kT[f"concentration={concentration}"][frame] = dict(sorted(polymer_size_by_kT[f"concentration={concentration}"][frame].items(), key=lambda item: float(item[0].split("=")[1])))

    # Sort frame
    polymer_size_by_kT[f"concentration={concentration}"] = dict(sorted(polymer_size_by_kT[f"concentration={concentration}"].items(), key=lambda item: float(item[0].split(" ")[1])))

    print(polymer_size_by_kT)

    with open("polymer_dist_by_kT_full_traj.json", "w") as json_file:
        json.dump(polymer_size_by_kT, json_file, indent=4)

    
    import matplotlib.pyplot as plt
    import numpy as np

    with open("polymer_dist_by_kT_full_traj.json", "r") as json_file:
        distribution = json.load(json_file)

    for concentration in distribution.keys():
        print(concentration)

        # Create plots
        fig, ax = plt.subplots(figsize=(7,5))

        for frame in distribution[concentration].keys():
            kT = []
            monomer_numbers = {}
            monomer_number_errors = {}

            for key in distribution[concentration][frame]:
                kT.append(float(key[3:]))
                all_monomers = sum(float(distribution[concentration][frame][key]["Average"][num_monomers]) for num_monomers in distribution[concentration][frame][key]["Average"].keys())

                for num_monomers in distribution[concentration][frame][key]["Average"].keys():
                    if num_monomers not in monomer_numbers.keys():
                        monomer_numbers[num_monomers] = []
                    monomer_numbers[num_monomers].append(float(distribution[concentration][frame][key]["Average"][num_monomers])/all_monomers)
                
                for num_monomers in distribution[concentration][frame][key]["Std_Dev"].keys():
                    if num_monomers not in monomer_number_errors.keys():
                        monomer_number_errors[num_monomers] = []
                    monomer_number_errors[num_monomers].append(float(distribution[concentration][frame][key]["Std_Dev"][num_monomers])/all_monomers)


            # Plot data
            for i, num_monomers in enumerate(monomer_numbers.keys()):
                if int(num_monomers.split(" ")[0]) == 2: # This only plots yield of two monomer units
                    ax.errorbar(1/np.array(kT)[::-1], np.array(monomer_numbers[num_monomers])[::-1], yerr=np.array(monomer_number_errors[num_monomers])[::-1], capsize=2.0, label=frame)

        # Axes Labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("1/kT")
        ax.set_ylabel("Fraction of Polymers")
        ax.set_title(f"Dimer Yield by kT for {concentration}")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"polymer_distribution_running_avg_{concentration}.svg")
        plt.close()
    



def plot_running_averages():
    import matplotlib.pyplot as plt
    import numpy as np
    with open("polymer_dist_by_kT_full_traj.json", "r") as json_file:
        distribution = json.load(json_file)

    for concentration in distribution.keys():
        # print(concentration)

        # Create plots
        fig, ax = plt.subplots(figsize=(7,5))

        kT = []
        for frame in distribution[concentration].keys():
            monomer_numbers = {}
            monomer_number_errors = {}

            for key in distribution[concentration][frame]:
                kT.append(float(key[3:]))

        kT = sorted(list(set(kT)))
        for temperature in kT:
            key = f"kT={temperature}"
            monomer_numbers = {}
            monomer_number_errors = {}
            for frame in distribution[concentration].keys():

                all_monomers = sum(float(distribution[concentration][frame][key]["Average"][num_monomers]) for num_monomers in distribution[concentration][frame][key]["Average"].keys())

                for num_monomers in distribution[concentration][frame][key]["Average"].keys():
                    if num_monomers not in monomer_numbers.keys():
                        monomer_numbers[num_monomers] = []
                    monomer_numbers[num_monomers].append(float(distribution[concentration][frame][key]["Average"][num_monomers])/all_monomers)
                
                for num_monomers in distribution[concentration][frame][key]["Std_Dev"].keys():
                    if num_monomers not in monomer_number_errors.keys():
                        monomer_number_errors[num_monomers] = []
                    monomer_number_errors[num_monomers].append(float(distribution[concentration][frame][key]["Std_Dev"][num_monomers])/all_monomers)

            timesteps = [int(frame.split(" ")[1])*1e5 for frame in distribution[concentration].keys()]
            # Plot data
            ax.errorbar(timesteps, np.array(monomer_numbers["2 panels"]), yerr=np.array(monomer_number_errors["2 panels"]), capsize=2.0, label=key)

        plt.axhline(y=0.5036304014446209, ls="--", color="red", label="yield=0.5")

        # Axes Labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Fraction of Polymers")
        ax.set_title(f"Dimer Yield by kT for {concentration}")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"polymer_distribution_running_avg_by_kT_{concentration}.png")
        plt.close()



def save_polymer_distribution_by_predicted_yield():
    """This only works for a single concentration, currently"""
    import signac
    from statistics import mean, stdev

    project = signac.get_project()
    # polymer_size_by_kT = {}
    polymer_size_by_predicted_yield = {}
    temperatures = set({})
    predicted_yields = set({})
    for job in project:
        temperatures.add(job.sp.kT)
        predicted_yields.add(job.sp.target_yield)

    for kT in temperatures:
        polymer_size_by_predicted_yield[f"kT={kT}"] = {}

        polymer_sizes = set({})
        max_polymer_size = 0
        for job in project.find_jobs({'kT': kT}):
            predicted_yield = job.sp.target_yield
            with job:
                with open("polymer_dist.json", "r") as json_file:
                    data = json.load(json_file)
                    if predicted_yield in polymer_size_by_predicted_yield[f"kT={kT}"].keys():
                        polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield][f"replica {job.sp.replica}"] = data["Averages"]
                    else:
                        polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield] = {f"replica {job.sp.replica}": data["Averages"]}
                    if len(data["Averages"]) > max_polymer_size:
                        max_polymer_size += 1
                        for polymer_size in data["Averages"]:
                            polymer_sizes.add(polymer_size)
        
        for predicted_yield in polymer_size_by_predicted_yield[f"kT={kT}"]:
            Data = {polymer_size: [] for polymer_size in polymer_sizes}
            for replica in polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield]:
                for polymer in Data.keys():
                    if polymer in polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield][replica].keys():
                        Data[polymer].append(polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield][replica][polymer])
                    else: 
                        Data[polymer].append(0)

            Average = {}
            Stdev = {}
            for polymer in Data:
                Average[polymer] = round(mean(Data[polymer]))
                if len(Data[polymer]) > 1:
                    Stdev[polymer] = stdev(Data[polymer])
                else:
                    Stdev[polymer] = 0
            polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield]["Average"] = Average
            polymer_size_by_predicted_yield[f"kT={kT}"][predicted_yield]["Std_Dev"] = Stdev

    print(polymer_size_by_predicted_yield)

    with open("polymer_dist_by_predicted_yield.json", "w") as json_file:
        json.dump(polymer_size_by_predicted_yield, json_file, indent=4, sort_keys=True)



def plot_by_predicted_yield():
    
    import matplotlib.pyplot as plt
    import numpy as np

    with open("polymer_dist_by_predicted_yield.json", "r") as json_file:
        distribution = json.load(json_file)

    # Create plots
    fig, ax = plt.subplots()
    
    for kT in distribution.keys():
        print(kT)
        predicted_yield = []
        monomer_numbers = {}
        monomer_number_errors = {}

        for key in distribution[kT]:
            predicted_yield.append(float(key))
            all_monomers = sum(float(distribution[kT][key]["Average"][num_monomers]) for num_monomers in distribution[kT][key]["Average"].keys())

            for num_monomers in  distribution[kT][key]["Average"].keys():
                if num_monomers not in monomer_numbers.keys():
                    monomer_numbers[num_monomers] = []
                monomer_numbers[num_monomers].append(float(distribution[kT][key]["Average"][num_monomers])/all_monomers)
            
            for num_monomers in  distribution[kT][key]["Std_Dev"].keys():
                if num_monomers not in monomer_number_errors.keys():
                    monomer_number_errors[num_monomers] = []
                monomer_number_errors[num_monomers].append(float(distribution[kT][key]["Std_Dev"][num_monomers])/all_monomers)

        
        for i, num_monomers in enumerate(monomer_numbers.keys()):
            if i == 1: # This only plots yield of two monomer units
                ax.errorbar(predicted_yield, np.array(monomer_numbers[num_monomers]), yerr=np.array(monomer_number_errors[num_monomers]), capsize=2.0, linestyle="", label=f"{kT}")

    ax.plot(np.arange(0,1,0.001), np.arange(0,1,0.001), linestyle="--", label="$y=x$")
    
    # Axes Labels
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Theoretical Dimer Yield")
    ax.set_ylabel("Simulated Dimer Yield")
    ax.set_title(f"Simulated Dimer Yield as a Function of Theoretical Dimer Yield")
    ax.legend()
    plt.savefig(f"yield_comparison_by_kT_concentration=0.001.png")
    plt.close()






def save_polymer_distribution_by_morse_strength():
    """This only works for a single concentration, currently"""
    import signac
    from statistics import mean, stdev

    project = signac.get_project()
    # polymer_size_by_kT = {}
    polymer_size_by_morse_strength = {}
    temperatures = set({})
    morse_strengths = set({})
    for job in project:
        temperatures.add(job.sp.kT)
        morse_strengths.add(mean(job.sp.d_colors))

    for kT in temperatures:
        polymer_size_by_morse_strength[f"kT={kT}"] = {}

        polymer_sizes = set({})
        max_polymer_size = 0
        for job in project.find_jobs({'kT': kT}):
            morse_strength = mean(job.sp.d_colors)
            with job:
                with open("polymer_dist.json", "r") as json_file:
                    data = json.load(json_file)
                    if morse_strength in polymer_size_by_morse_strength[f"kT={kT}"].keys():
                        polymer_size_by_morse_strength[f"kT={kT}"][morse_strength][f"replica {job.sp.replica}"] = data["Averages"]
                    else:
                        polymer_size_by_morse_strength[f"kT={kT}"][morse_strength] = {f"replica {job.sp.replica}": data["Averages"]}
                    if len(data["Averages"]) > max_polymer_size:
                        max_polymer_size += 1
                        for polymer_size in data["Averages"]:
                            polymer_sizes.add(polymer_size)
        
        for morse_strength in polymer_size_by_morse_strength[f"kT={kT}"]:
            Data = {polymer_size: [] for polymer_size in polymer_sizes}
            for replica in polymer_size_by_morse_strength[f"kT={kT}"][morse_strength]:
                for polymer in Data.keys():
                    if polymer in polymer_size_by_morse_strength[f"kT={kT}"][morse_strength][replica].keys():
                        Data[polymer].append(polymer_size_by_morse_strength[f"kT={kT}"][morse_strength][replica][polymer])
                    else: 
                        Data[polymer].append(0)

            Average = {}
            Stdev = {}
            for polymer in Data:
                Average[polymer] = round(mean(Data[polymer]))
                if len(Data[polymer]) > 1:
                    Stdev[polymer] = stdev(Data[polymer])
                else:
                    Stdev[polymer] = 0
            polymer_size_by_morse_strength[f"kT={kT}"][morse_strength]["Average"] = Average
            polymer_size_by_morse_strength[f"kT={kT}"][morse_strength]["Std_Dev"] = Stdev

    print(polymer_size_by_morse_strength)

    with open("polymer_dist_by_morse_strength.json", "w") as json_file:
        json.dump(polymer_size_by_morse_strength, json_file, indent=4, sort_keys=True)



def plot_by_morse_strength():
    
    import matplotlib.pyplot as plt
    import numpy as np

    with open("polymer_dist_by_morse_strength.json", "r") as json_file:
        distribution = json.load(json_file)

    # Create plots
    fig, ax = plt.subplots()
    
    for kT in distribution.keys():
        print(kT)
        morse_strength = []
        monomer_numbers = {}
        monomer_number_errors = {}

        for key in distribution[kT]:
            morse_strength.append(float(key))
            all_monomers = sum(float(distribution[kT][key]["Average"][num_monomers]) for num_monomers in distribution[kT][key]["Average"].keys())

            for num_monomers in  distribution[kT][key]["Average"].keys():
                if num_monomers not in monomer_numbers.keys():
                    monomer_numbers[num_monomers] = []
                monomer_numbers[num_monomers].append(float(distribution[kT][key]["Average"][num_monomers])/all_monomers)
            
            for num_monomers in  distribution[kT][key]["Std_Dev"].keys():
                if num_monomers not in monomer_number_errors.keys():
                    monomer_number_errors[num_monomers] = []
                monomer_number_errors[num_monomers].append(float(distribution[kT][key]["Std_Dev"][num_monomers])/all_monomers)


        # Plot data

        yields_and_morse_strengths = sorted([(0.09597747029303343,[7.205025038459917,7.204855518668997,7.204855518668997]),
                                            (0.2039951246534103,[7.595931890618219,7.595718108873248,7.595718108873248]),
                                            (0.3007093099485724,[7.90311610380892,7.903094501433354,7.903094501433354]),
                                            (0.39712898280277736,[8.150645036335282,8.150769711321391,8.150769711321391]),
                                            (0.4965295150383015,[8.374183551473745,8.37458317177438,8.37458317177438]),
                                            (0.596295007651516,[8.678999427769229,8.679225798219303,8.679225798219303]),
                                            (0.4185754642946019,[8.189620803378885,8.189341407082665,8.189328406568665]),
                                            (0.71157346935909,[9.028693357747912,9.028277222569272,9.027931728575036]),
                                            (0.7836756853444697,[9.273448930071483,9.273029694935518,9.272601961204227]),
                                            (0.837173263652429,[9.483634829555603,9.483089578340095,9.482561248232166]),
                                            (0.8738647986876894,[9.733381258425391,9.732870757360502,9.732183682701613])])
        
        theoretical_yields = []
        morse_strength_values = []

        for theoretical_yield, strengths in yields_and_morse_strengths:
            theoretical_yields.append(theoretical_yield)
            morse_strength_values.append(np.mean(strengths))

        # ax.plot(morse_strength_values, theoretical_yields, label="theoretical yields")
        
        for i, num_monomers in enumerate(monomer_numbers.keys()):
            if i == 1: # This only plots yield of two monomer units
                ax.errorbar(theoretical_yields, np.array(monomer_numbers[num_monomers]), yerr=np.array(monomer_number_errors[num_monomers]), capsize=2.0, linestyle="", label=f"kT={kT}")

    ax.plot(np.arange(0,1,0.001), np.arange(0,1,0.001), linestyle="--", label="$y=x$")
    
    # Axes Labels
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Theoretical Dimer Yield")
    ax.set_ylabel("Simulated Dimer Yield")
    ax.set_title(f"Simulated Dimer Yield as a Function of Theoretical Dimer Yield at $k_b T = 1.0$")
    ax.legend()
    plt.savefig(f"yield_comparison_concentration=0.001.png")
    plt.close()



def save_polymer_distribution_by_N():
    """This only works for a single kT, currently"""
    import signac
    from statistics import mean, stdev

    project = signac.get_project()
    # polymer_size_by_kT = {}
    polymer_size_by_total_N = {}
    concentrations = set({})
    total_Ns = set({})
    for job in project:
        concentrations.add(job.sp.concentration)
        total_Ns.add(2 * job.sp.Nx * job.sp.Ny * job.sp.Nz) # NOTE: This is for a unitcell size of 2.

    for concentration in concentrations:
        polymer_size_by_total_N[f"concentration={concentration}"] = {}

        polymer_sizes = set({})
        max_polymer_size = 0
        for job in project.find_jobs({'concentration': concentration}):
            total_N = 2 * job.sp.Nx * job.sp.Ny * job.sp.Nz
            with job:
                with open("polymer_dist.json", "r") as json_file:
                    data = json.load(json_file)
                    if total_N in polymer_size_by_total_N[f"concentration={concentration}"].keys():
                        polymer_size_by_total_N[f"concentration={concentration}"][total_N][f"replica {job.sp.replica}"] = data["Averages"]
                    else:
                        polymer_size_by_total_N[f"concentration={concentration}"][total_N] = {f"replica {job.sp.replica}": data["Averages"]}
                    if len(data["Averages"]) > max_polymer_size:
                        max_polymer_size += 1
                        for polymer_size in data["Averages"]:
                            polymer_sizes.add(polymer_size)
        
        for total_N in polymer_size_by_total_N[f"concentration={concentration}"]:
            Data = {polymer_size: [] for polymer_size in polymer_sizes}
            for replica in polymer_size_by_total_N[f"concentration={concentration}"][total_N]:
                for polymer in Data.keys():
                    if polymer in polymer_size_by_total_N[f"concentration={concentration}"][total_N][replica].keys():
                        Data[polymer].append(polymer_size_by_total_N[f"concentration={concentration}"][total_N][replica][polymer])
                    else: 
                        Data[polymer].append(0)

            Average = {}
            Stdev = {}
            for polymer in Data:
                Average[polymer] = round(mean(Data[polymer]))
                if len(Data[polymer]) > 1:
                    Stdev[polymer] = stdev(Data[polymer])
                else:
                    Stdev[polymer] = 0
            polymer_size_by_total_N[f"concentration={concentration}"][total_N]["Average"] = Average
            polymer_size_by_total_N[f"concentration={concentration}"][total_N]["Std_Dev"] = Stdev

    print(polymer_size_by_total_N)

    with open("polymer_dist_by_total_N.json", "w") as json_file:
        json.dump(polymer_size_by_total_N, json_file, indent=4, sort_keys=True)



def plot_by_N():
    
    import matplotlib.pyplot as plt
    import numpy as np

    with open("polymer_dist_by_total_N.json", "r") as json_file:
        distribution = json.load(json_file)

    for concentration in distribution.keys():
        print(concentration)
        total_N = []
        monomer_numbers = {}
        monomer_number_errors = {}

        for key in distribution[concentration]:
            total_N.append(float(key))
            all_monomers = sum(float(distribution[concentration][key]["Average"][num_monomers]) for num_monomers in distribution[concentration][key]["Average"].keys())

            for num_monomers in  distribution[concentration][key]["Average"].keys():
                if num_monomers not in monomer_numbers.keys():
                    monomer_numbers[num_monomers] = []
                monomer_numbers[num_monomers].append(float(distribution[concentration][key]["Average"][num_monomers])/all_monomers)
            
            for num_monomers in  distribution[concentration][key]["Std_Dev"].keys():
                if num_monomers not in monomer_number_errors.keys():
                    monomer_number_errors[num_monomers] = []
                monomer_number_errors[num_monomers].append(float(distribution[concentration][key]["Std_Dev"][num_monomers])/all_monomers)

        # Create plots
        fig, ax = plt.subplots()

        # Plot data
        for i, num_monomers in enumerate(monomer_numbers.keys()):
            if i <= 1: # This only plots yield of one and two monomer units
                ax.errorbar(total_N, np.array(monomer_numbers[num_monomers]), yerr=np.array(monomer_number_errors[num_monomers]), capsize=2.0, label=f"{num_monomers.split(" ")[0]} monomers")

        # Axes Labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("N")
        ax.set_ylabel("Fraction of Polymers")
        ax.set_title(f"Polymer Size Distributions by N for Target Yield of 0.5")
        ax.legend()
        plt.savefig(f"polymer_distribution_by_N_{concentration}.svg")
        plt.close()


if __name__ == "__main__":
    globals()[sys.argv[1]]()