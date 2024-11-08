from visuals_fabrics.visual_creator import VisualCreator

if __name__ == "__main__":
    creator = VisualCreator(debug = False)
    creator.set_config_files(
        "data/invite_trajectories.npy",
        "configs/invite_config.yaml",
        "configs/invite_setup.yaml",
    )
    creator.load_configs()
    #creator.generate_trajectories()
    width = 120
    height = 170
    creator.init_figure(width, height)
    creator.load_setup_data()
    creator.create_visual("outputs/invite")

