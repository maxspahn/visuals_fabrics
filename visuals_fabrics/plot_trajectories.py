import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(file_path: str, limits: dict, obstacles: list):
    trajectories = np.load(file_path, allow_pickle=True)
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    N = len(trajectories)
    blue_colors = plt.cm.Blues(np.linspace(0.2, 1, N))

    for i, trajectory in enumerate(trajectories):
        q = trajectory[np.linalg.norm(trajectory, axis=1) > 1e-5]
        ax.plot(q[:, 0], q[:, 1], color=blue_colors[i], linewidth=3)
    plt.xlim(limits["lower_limits"][0] - 0.1, limits["upper_limits"][0] - 0.1)
    plt.ylim(limits["lower_limits"][1] - 0.1, limits["upper_limits"][1] - 0.1)
    # plot box for limits
    #ax.plot([limits["lower_limits"][0], limits["lower_limits"][0]], [limits["lower_limits"][1], limits["upper_limits"][1]], 'k', color='white')
    #ax.plot([limits["upper_limits"][0], limits["upper_limits"][0]], [limits["lower_limits"][1], limits["upper_limits"][1]], 'k', color='white')
    #ax.plot([limits["lower_limits"][0], limits["upper_limits"][0]], [limits["lower_limits"][1], limits["lower_limits"][1]], 'k', color='white')
    #ax.plot([limits["lower_limits"][0], limits["upper_limits"][0]], [limits["upper_limits"][1], limits["upper_limits"][1]], 'k', color='white')
    
    for obstacle in obstacles:
        if obstacle.position()[0] > 100:
            continue
        if obstacle.type() == 'sphere':
            patch = plt.Circle(
                obstacle.position(),
                radius=obstacle.radius(),
            )
        elif obstacle.type() == 'box':
            box_obstacle_center = [
                obstacle.position()[0] - 0.5 * obstacle.size()[0],
                obstacle.position()[1] - 0.5 * obstacle.size()[1]
            ]
            patch = plt.Rectangle(
                box_obstacle_center,
                width=obstacle.size()[0],
                height=obstacle.size()[1],
            )
        patch.set_linewidth(2)
        patch.set_edgecolor('none')
        patch.set_facecolor('red')
        patch.set_alpha(0.5)
        ax.add_patch(patch)
    # Plot goal
    goal_patch = plt.Circle(
        q[-1, :],
        radius=0.1,
    )
    goal_patch.set_edgecolor('none')
    goal_patch.set_facecolor('green')
    goal_patch.set_alpha(0.7)
    ax.add_patch(goal_patch)
    start_patch = plt.Circle(
        q[0, :],
        radius=0.1,
    )
    start_patch.set_edgecolor('none')
    start_patch.set_facecolor('blue')
    start_patch.set_alpha(0.7)
    ax.add_patch(start_patch)

    ax.axis('equal')
    plt.axis('off')
    # save as vector graphics

    plt.savefig(file_path.split('.')[0] + ".svg", bbox_inches='tight', pad_inches=0) 

