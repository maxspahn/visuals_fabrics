from visuals_fabrics.visual_creator import VisualCreator

if __name__ == "__main__":
    creator = VisualCreator(debug = False)
    creator.set_config_files(
        "data/title_trajectories.npy",
        "configs/title_config.yaml",
        "configs/title_setup.yaml",
    )
    creator.load_configs()
    creator.generate_trajectories()
    width = 320
    height = 180
    creator.init_figure(width, height)
    creator.load_setup_data()
    creator.create_visual("outputs/title")

