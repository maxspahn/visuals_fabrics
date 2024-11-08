from visuals_fabrics.visual_creator import VisualCreator

if __name__ == "__main__":
    creator = VisualCreator(debug = False)
    creator.set_config_files(
        "data/cover_trajectories.npy",
        "configs/cover_config.yaml",
        "configs/cover_setup.yaml",
    )
    creator.load_configs()
    #creator.generate_trajectories()
    width = 172+172+9.22
    height = 244
    creator.init_figure(width, height)
    creator.load_setup_data()
    if creator.debug:
        creator.ax.plot([0.461, 0.461], [-12.0, 12.0], color='white')
        creator.ax.plot([-0.461, -0.461], [-12.0, 12.0], color='white')
        creator.ax.plot([-17.661, -17.661], [-12.2, 12.2], color='white')
        creator.ax.plot([17.661, 17.661], [-12.2, 12.2], color='white')
        creator.ax.plot([-17.661, 17.661], [12.2, 12.2], color='white')
        creator.ax.plot([-17.661, 17.661], [-12.2, -12.2], color='white')

        creator.ax.plot([-17.461, -17.461], [-12.0, 12.0], color='white')
        creator.ax.plot([17.461, 17.461], [-12.0, 12.0], color='white')
        creator.ax.plot([-17.461, 17.461], [12.0, 12.0], color='white')
        creator.ax.plot([-17.461, 17.461], [-12.0, -12.0], color='white')
    creator.create_visual("outputs/cover")

