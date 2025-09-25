import numpy as np
import signac
from flow import FlowProject
from flow import directives
import flow.environments
import json
import os
import gsd.hoomd

class Project(FlowProject):
    pass

@Project.label
def dumped(job):
    if job.isfile("dump.gsd"):
        with job:
            with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
                if len(gsd_file) == 1600:
                    return True
    return False

@Project.label
def rendered(job):
    return job.isfile("render.png")

@Project.pre(dumped)
@Project.post(rendered)
@Project.operation
def render(job):
    import numpy as np
    import gsd.hoomd
    import fresnel
    from random import randint

    # Create an image
    with job:
        # Read the trajectory
        with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
            snap = gsd_file[-1]

        box = snap.configuration.box

        # Colour particles by type
        N = snap.particles.N
        individual_particle_types = snap.particles.types
        particle_types = snap.particles.typeid
        colors = np.empty((N, 3))
        colors_by_type = []
        for i in range(len(individual_particle_types)):
            color = '#%06X' % randint(0, 0xFFFFFF)
            color = list(int(color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            colors_by_type.append(color)

        radii = np.zeros((snap.particles.typeid.shape))
        #['A', 'B', 'AM', 'AP1', 'AP2', 'AP3', 'BM', 'BP1', 'BP2', 'BP3']
        radii[np.isin(snap.particles.typeid, [2, 6])] = 1.0
        radii[np.isin(snap.particles.typeid, [3,4,5,7,8,9])] = 0.3

        # Color by typeid
        for i in range(len(individual_particle_types)):
            colors[particle_types == i] = fresnel.color.linear(colors_by_type[i])

        # Set the scene
        scene = fresnel.Scene(
            camera=fresnel.camera.Orthographic(
                position=(100, 100, 100),
                look_at=(0, 0, 0),
                up=(0, 1, 0),
                height=100,
            )
        )

        # Spheres for every particle in the system
        geometry = fresnel.geometry.Sphere(scene, N=N, radius=radii)
        geometry.position[:] = snap.particles.position
        geometry.material = fresnel.material.Material(roughness=0.1)
        geometry.outline_width = 0.05

        # use color instead of material.color
        geometry.material.primitive_color_mix = 1.0
        geometry.color[:] = fresnel.color.linear(colors)

        # create box in fresnel
        fresnel.geometry.Box(scene, box, box_radius=.07)

        # Render the system
        scene.lights = fresnel.light.lightbox()
        # out = fresnel.pathtrace(scene, light_samples=10, w=1380, h=1380)

        # Save image to file
        out = fresnel.preview(scene, w=3600, h=2220)
        print(out[:].shape)
        print(out[:].dtype)

        import PIL
        image = PIL.Image.fromarray(out[:], mode='RGBA')
        image.save('render.png')


@Project.label
def plotted(job):
    return (job.isfile("plots/pressure.png") 
            and job.isfile("plots/potential_energy.png") 
            and job.isfile("plots/translational_kinetic_energy.png") 
            and job.isfile("plots/rotational_kinetic_energy.png"))

@Project.pre(dumped)
@Project.post(plotted)
@Project.operation
def plot_quantities(job):
    import matplotlib.pyplot as plt
    import h5py
    
    with job: 

        data = h5py.File(name='dump_log.h5', mode='r')
        timestep = data['hoomd-data/Simulation/timestep'][:]
        pressure = data['/hoomd-data/md/compute/ThermodynamicQuantities/pressure'][:]
        potential_energy = data['/hoomd-data/md/compute/ThermodynamicQuantities/potential_energy'][:]
        translational_kinetic_energy = data['/hoomd-data/md/compute/ThermodynamicQuantities/translational_kinetic_energy'][:]
        rotational_kinetic_energy = data['/hoomd-data/md/compute/ThermodynamicQuantities/rotational_kinetic_energy'][:]

        if os.path.exists("plots") == False: 
            os.mkdir("plots")
        
        plt.close('all')
        fig, ax = plt.subplots()

        # Pressure
        ax.plot(timestep, pressure)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="pressure", title="Pressure")
        fig.savefig("plots/pressure.png")

        # Potential energy
        ax.clear()
        ax.plot(timestep, potential_energy)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="potential energy", title="Potential Energy")
        fig.savefig("plots/potential_energy.png")

        # Translational kinetic energy
        ax.clear()
        ax.plot(timestep, translational_kinetic_energy)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="translational kinetic energy", title="Translational Kinetic Energy")
        fig.savefig("plots/translational_kinetic_energy.png")

        # Rotational Kinetic Energy
        ax.clear()
        ax.plot(timestep, rotational_kinetic_energy)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="rotational kinetic energy", title="Rotational Kinetic Energy")
        fig.savefig("plots/rotational_kinetic_energy.png")

        
@Project.label
def polymers_identified(job):
    return(job.isfile("polymers/raw_point.png")
            and job.isfile("polymers/polymers.png")
            and job.isfile("polymers/polymer_size_dist.png")
            and job.isfile("polymer_dist.json"))

# @Project.pre.after(plot_quantities)
@Project.pre(dumped)
@Project.post(polymers_identified)
@Project.operation
def identify_polymers(job):
    import matplotlib.pyplot as plt
    import gsd.hoomd
    import freud

    with job:

        plt.close('all')

        if os.path.exists(job.fn("polymers")) == False: 
            os.mkdir("polymers")

        # Read equilibrated system from GSD file
        with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
            frames = gsd_file[-10:]
            polymer_sizes_by_frame = {}
            for i, frame in enumerate(frames):
                current_frame = int(frame.configuration.step / 1e5)
                positions = []
                for index in range(frame.particles.N):
                    if frame.particles.typeid[index] == 0 or frame.particles.typeid[index] == 1:
                        positions.append(frame.particles.position[index])
                
                box = freud.box.Box.from_box(frame.configuration.box, dimensions = 3)
                system = freud.AABBQuery(box, np.array(positions))

                # Plot and save central particle positions before clustering. 
                if i == len(frames) - 1:
                    plt.clf()
                    fig = plt.figure()
                    system.plot(ax=fig.add_subplot(projection='3d'), s=10)
                    # plt.title('Raw points before clustering', fontsize=20)
                    plt.gca().tick_params(axis='both', which='both', labelsize=7, size=4)
                    plt.savefig("polymers/raw_point.png")

                # Identify polymers
                cl = freud.cluster.Cluster()
                cl.compute(system, neighbors={"mode": "ball", "r_max": 2.5})
                print(cl.cluster_idx)
                print(cl.num_clusters)

                # Plot polymers
                if i == len(frames) - 1:
                    plt.clf()
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for cluster_id in range(cl.num_clusters):
                        cluster_system = freud.AABBQuery(system.box, system.points[cl.cluster_keys[cluster_id]])
                        cluster_system.plot(ax=ax, s=10, label="Cluster {}".format(cluster_id))
                        # print("There are {} points in cluster {}.".format(len(cl.cluster_keys[cluster_id]), cluster_id))

                    # ax.set_title('Clusters identified', fontsize=20)
                    ax.legend(loc='right', fontsize=4)
                    ax.tick_params(axis='both', which='both', labelsize=7, size=4)
                    plt.savefig("polymers/polymers.png")

                # Calculate occurences of polymer sizes
                lengths = []
                polymers = []

                for cluster_id in range(cl.num_clusters):
                    if len(cl.cluster_keys[cluster_id]) > len(lengths):
                        lengths.extend(
                            [0] * (len(cl.cluster_keys[cluster_id]) - len(lengths))
                            )
                    lengths[len(cl.cluster_keys[cluster_id]) - 1] += 1
                    
                    polymers += [len(cl.cluster_keys[cluster_id])]

                print(lengths)
                print(polymers)

                polymer_sizes = {}
                for length in range(len(lengths)):
                    polymer_sizes[f"{str(length + 1)} panels"] = lengths[length]
                polymer_sizes_by_frame[f"Frame {int(frame.configuration.step / 1e5)}"] = polymer_sizes

            # Save polymer size distribution in a json file
            polymer_size_averages = {}
            for key in polymer_sizes_by_frame:
                for subkey in polymer_sizes_by_frame[key]:
                    if subkey in polymer_size_averages.keys():
                        polymer_size_averages[subkey] += np.array([polymer_sizes_by_frame[key][subkey],1])
                    else: 
                        polymer_size_averages[subkey] = np.array([polymer_sizes_by_frame[key][subkey],1])
            for key in polymer_size_averages:
                polymer_size_averages[key] = round(polymer_size_averages[key][0]/polymer_size_averages[key][1])
            polymer_sizes_by_frame["Averages"] = polymer_size_averages

            with open("polymer_dist.json", "w") as json_file:
                json.dump(polymer_sizes_by_frame, json_file, indent=4, sort_keys=True)

            # Plot polymer size distribution (procedure from https://matplotlib.org/stable/gallery/statistics/barchart_demo.html)
            plt.clf()
            fig, ax1 = plt.subplots(figsize=(9, 7))  # Create the figure
            fig.subplots_adjust(left=0.215, right=0.88)
            fig.canvas.manager.set_window_title('Polymer Size Distribution')

            pos = np.array(np.arange(len(polymer_size_averages))) + 1

            rects = ax1.barh(pos, [polymer_size_averages[key] for key in sorted(polymer_size_averages)],
                                align='center',
                                height=0.5,
                                tick_label=[str(p) + " Monomer" + ("s" if p != 1 else "") for p in pos])

            ax1.set_title(f"Polymer Size Distribution")
            ax1.set_ylabel("Polymer Size")

            ax1.set_xlim([0, 100])
            ax1.xaxis.grid(True, linestyle='--', which='major',
                            color='grey', alpha=.25)

            # Write the sizes inside each bar to aid in interpretation
            rect_labels = []
            for rect in rects:
                width = rect.get_width()
                # The bars aren't wide enough to print the ranking inside
                if width < 40:
                    # Shift the text to the right side of the right edge
                    xloc = 5
                    # Black against white background
                    clr = 'black'
                    align = 'left'
                else:
                    # Shift the text to the left side of the right edge
                    xloc = -5
                    # White on magenta
                    clr = 'white'
                    align = 'right'

                # Center the text vertically in the bar
                yloc = rect.get_y() + rect.get_height() / 2
                label = ax1.annotate(
                    width, xy=(width, yloc), xytext=(xloc, 0),
                    textcoords="offset points",
                    horizontalalignment=align, verticalalignment='center',
                    color=clr, weight='bold', clip_on=True)
                rect_labels.append(label)

            plt.savefig("polymers/polymer_size_dist.png")



@Project.label
def polymers_identified_full_traj(job):
    return job.isfile("polymer_dist_full_traj.json")


@Project.pre(dumped)
@Project.post(polymers_identified_full_traj)
@Project.operation
def identify_polymers_full_traj(job):
    import matplotlib.pyplot as plt
    import gsd.hoomd
    import freud

    with job:

        # Read equilibrated system from GSD file
        with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
            skip = 30
            frames = gsd_file[::skip]
            polymer_sizes_by_frame = {}
            for i, frame in enumerate(frames):
                current_frame = int(frame.configuration.step / 1e5)
                positions = []
                for index in range(frame.particles.N):
                    if frame.particles.typeid[index] == 0 or frame.particles.typeid[index] == 1:
                        positions.append(frame.particles.position[index])
                
                box = freud.box.Box.from_box(frame.configuration.box, dimensions = 3)
                system = freud.AABBQuery(box, np.array(positions))

                # Identify polymers
                cl = freud.cluster.Cluster()
                cl.compute(system, neighbors={"mode": "ball", "r_max": 2.5})

                # Calculate occurences of polymer sizes
                lengths = []
                polymers = []

                for cluster_id in range(cl.num_clusters):
                    if len(cl.cluster_keys[cluster_id]) > len(lengths):
                        lengths.extend(
                            [0] * (len(cl.cluster_keys[cluster_id]) - len(lengths))
                            )
                    lengths[len(cl.cluster_keys[cluster_id]) - 1] += 1
                    
                    polymers += [len(cl.cluster_keys[cluster_id])]

                polymer_sizes = {}
                for length in range(len(lengths)):
                    polymer_sizes[f"{str(length + 1)} panels"] = lengths[length]
                polymer_sizes_by_frame[f"Frame {int(frame.configuration.step / 1e5)}"] = polymer_sizes

            # Save polymer size distribution in a json file
            with open("polymer_dist_full_traj.json", "w") as json_file:
                json.dump(polymer_sizes_by_frame, json_file, indent=4, sort_keys=True)


# @Project.label
# def polymers_identified_full_dump2(job):
#     return job.isfile("polymer_dist_full_dump2.json")

# @Project.post(polymers_identified_full_traj)
# @Project.operation
# def identify_polymers_full_dump2(job):
#     import matplotlib.pyplot as plt
#     import gsd.hoomd
#     import freud

#     with job:

#         # Read equilibrated system from GSD file
#         with gsd.hoomd.open(name="dump2.gsd", mode="r") as gsd_file:
#             skip = 30
#             frames = gsd_file[::skip]
#             polymer_sizes_by_frame = {}
#             for i, frame in enumerate(frames):
#                 current_frame = int(frame.configuration.step / 1e5)
#                 positions = []
#                 for index in range(frame.particles.N):
#                     if frame.particles.typeid[index] == 0 or frame.particles.typeid[index] == 1:
#                         positions.append(frame.particles.position[index])
                
#                 box = freud.box.Box.from_box(frame.configuration.box, dimensions = 3)
#                 system = freud.AABBQuery(box, np.array(positions))

#                 # Identify polymers
#                 cl = freud.cluster.Cluster()
#                 cl.compute(system, neighbors={"mode": "ball", "r_max": 2.5})

#                 # Calculate occurences of polymer sizes
#                 lengths = []
#                 polymers = []

#                 for cluster_id in range(cl.num_clusters):
#                     if len(cl.cluster_keys[cluster_id]) > len(lengths):
#                         lengths.extend(
#                             [0] * (len(cl.cluster_keys[cluster_id]) - len(lengths))
#                             )
#                     lengths[len(cl.cluster_keys[cluster_id]) - 1] += 1
                    
#                     polymers += [len(cl.cluster_keys[cluster_id])]

#                 polymer_sizes = {}
#                 for length in range(len(lengths)):
#                     polymer_sizes[f"{str(length + 1)} panels"] = lengths[length]
#                 polymer_sizes_by_frame[f"Frame {int(frame.configuration.step / 1e5)}"] = polymer_sizes

#             # Save polymer size distribution in a json file
#             with open("polymer_dist_full_dump2.json", "w") as json_file:
#                 json.dump(polymer_sizes_by_frame, json_file, indent=4, sort_keys=True)



if __name__ == '__main__':
    Project().main()