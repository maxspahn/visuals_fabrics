from visuals_fabrics.visual_creator import VisualCreator

if __name__ == "__main__":
    creator = VisualCreator(debug = False)
    creator.set_config_files(
        "data/invite_small_trajectories.npy",
        "configs/invite_small_config.yaml",
        "configs/invite_small_setup.yaml",
    )
    creator.load_configs()
    #creator.generate_trajectories()
    width = 64
    height = 244
    creator.init_figure(width, height)
    creator.load_setup_data()
    creator.create_visual("outputs/invite_small")

