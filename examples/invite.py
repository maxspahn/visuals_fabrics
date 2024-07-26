# pylint: disable=import-outside-toplevel
from copy import deepcopy
import sys
from typing import List
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
import numpy as np
import yaml

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, bbox_artist


from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from forwardkinematics.planarFks.pointFk import PointFk
from planarenvs.sensors.full_sensor import FullSensor

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# This example showcases the psedo-sensor and requires goals and obstacles.
# As a result, it requires the optional package motion_planning_scenes.

# Consider installing it with `poetry install -E scenes`.
obstacle_color = 'none'
obstacle_edge_color = 'none'

CONFIG_FILE = "invite_config.yaml"
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)
    CONFIG_PROBLEM = config['problem']
    CONFIG_FABRICS = config['fabrics']

DT = 0.01

default_goal = GoalComposition(name="goal", content_dict=CONFIG_PROBLEM["goal"]['goal_definition'])
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [0.0, 6, 0.0],
        "length": 5.0,
        "width": 1.0,
    },
}
obst_0 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [0.0, 3.0, 0.0],
        "length": 8.0,
        "width": 1.0,
    },
}
obst_1 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [0.0, -0.5, 0.0],
        "length": 3.5,
        "width": 2.0,
    },
}
obst_2 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [0, -6, 0.0],
        "length": 3.5,
        "width": 1,
    },
}
obst_3 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
obst_dict = {
    "type": "sphere",
    "geometry": {
        "position": [-6, 7.5, 0.0],
        "radius": 1
    },
}
obst_4 = SphereObstacle(name="simpleBox", content_dict=obst_dict)
LIMITS = {
    "high": np.array(CONFIG_PROBLEM['joint_limits']['upper_limits']),
    "low": np.array(CONFIG_PROBLEM['joint_limits']['lower_limits']),
}

user_defined_initial_positions = [
    np.array([-2.7, -5]),
    np.array([-3.3, -1.0]),
]
user_defined_goal_positions = [
    np.array([5, 1.0]),
    np.array([3.5, 6.7]),
]

def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    degrees_of_freedom = 2
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    limit_geometry: str = (
        "-10.1 / (x ** 1) * xdot ** 2"
    )
    limit_finsler: str = (
        "1.0/(x**2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
    )
    damper_beta: str = (
        "0.5 * (ca.tanh(-0.5 * (ca.norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + ca.fmax(0, sym('a_ex') - sym('a_le'))"
    )
    damper_eta: str = (
        "0.5 * (ca.tanh(-0.9 * (1 - 1/2) * ca.dot(xdot, xdot) - 0.5) + 1)"
    )
    damper_beta = "0"
    damper_eta = "1"
    forward_kinematics = PointFk()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        limit_geometry=limit_geometry,
        limit_finsler=limit_finsler,
        damper_beta=damper_beta,
        damper_eta=damper_eta,
    )
    collision_links = [1]
    # The planner hides all the logic behind the function set_components.
    # convert LIMITS to 2d array
    """
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
        limits=limits_as_array,
    )
    """
    planner.load_fabrics_configuration(CONFIG_FABRICS)
    planner.load_problem_configuration(CONFIG_PROBLEM)
    planner.concretize()
    return planner


def run_point_robot(
    planner: ParameterizedFabricPlanner,
    n_steps: int = 1000,
    render: bool = False,
    initial_position: np.ndarray = np.zeros(2),
    initial_velocity: np.ndarray = np.zeros(2),
    goal: GoalComposition = default_goal,
):
    env = PointRobotAccEnv(render=render, dt=DT)
    env.reset_limits(pos=LIMITS)
    ob = env.reset(pos=initial_position, vel=initial_velocity)


    env.add_obstacle(obst_0)
    env.add_obstacle(obst_1)
    env.add_obstacle(obst_2)
    env.add_obstacle(obst_3)
    env.add_obstacle(obst_4)
    env.add_goal(goal.sub_goals()[0])
    full_sensor = FullSensor(
        ['position', 'weight'],
        ['position', 'size'],
        variance=0.0
    )
    env.add_sensor(full_sensor)
    ob, *_ = env.step(np.zeros(2))

    print("Starting episode")
    observation_history = []
    q = np.zeros((n_steps, 2))
    qdot = np.zeros((n_steps, 2))
    arguments = get_arguments(goal)


    for i in range(n_steps):
        q[i, :] = ob['joint_state']['position']
        qdot[i, :] = ob['joint_state']['velocity']
        action = time_variant_action(env.t())
        arguments['q'] = ob['joint_state']['position']
        arguments['qdot'] = ob['joint_state']['velocity']
        action = planner.compute_action(**arguments)
        ob, reward, terminated, info = env.step(action)
        if terminated:
            return q
    print(arguments['x_goal_0'])
    print(q[0, :])
    return q

def break_condition(q, action, goal):
    outside_limits = q[0] > LIMITS["high"][0] or q[0] < LIMITS["low"][0] or q[1] > LIMITS["high"][1] or q[1] < LIMITS["low"][1]
    low_action_magnitude = np.linalg.norm(action) < 1e-4
    goal_reached = np.linalg.norm(q - goal.primary_goal().position()) < goal.primary_goal().epsilon()
    if low_action_magnitude:
        print("Low action magnitude")
    if outside_limits:
        print("Outside limits")
    if goal_reached:
        print("Goal reached")
    return outside_limits or low_action_magnitude or goal_reached

def position_in_collision(goal_location):

    return (
            np.linalg.norm(goal_location - obst_0.position()[0:2]) < goal.primary_goal().epsilon() + obst_0.size()[0] or
            np.linalg.norm(goal_location - obst_1.position()[0:2]) < goal.primary_goal().epsilon() + obst_1.size()[0] or
            np.linalg.norm(goal_location - obst_2.position()[0:2]) < goal.primary_goal().epsilon() + obst_2.size()[0] or
            np.linalg.norm(goal_location - obst_3.position()[0:2]) < goal.primary_goal().epsilon() + obst_3.size()[0]
    )

def compose_goal(goal_location):
    goal_dict = deepcopy(CONFIG_PROBLEM["goal"]['goal_definition'])
    goal_dict['subgoal0']['desired_position'] = goal_location.tolist()
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    return goal

def get_arguments(goal):
    arguments = {
        'x_goal_0': goal.primary_goal().position(),
        'weight_goal_0': goal.primary_goal().weight(),
        'radius_body_1': 0.1,
        'x_obst_0': obst_4.position(),
        'radius_obst_0': obst_4.size(),
        'x_obst_1': obst_0.position(),
        'sizes_obst_1': obst_0.size(),
        'x_obst_2': obst_1.position(),
        'sizes_obst_2': obst_1.size(),
        'x_obst_3': obst_2.position(),
        'sizes_obst_3': obst_2.size(),
        'x_obst_4': obst_3.position(),
        'sizes_obst_4': obst_3.size(),
    }
    return arguments

def run_point_robot_rollout(
    planner: ParameterizedFabricPlanner,
    n_steps: int = 1000,
    render: bool = False,
    initial_position: np.ndarray = np.zeros(2),
    initial_velocity: np.ndarray = np.zeros(2),
    goal: GoalComposition = default_goal,
):
    q = np.zeros((n_steps, 2))
    qdot = np.zeros((n_steps, 2))
    q[0, :] = initial_position
    qdot[0, :] = initial_velocity
    arguments = get_arguments(goal)

    for i in range(n_steps-1):
        arguments['q'] = q[i, :]
        arguments['qdot'] = qdot[i, :]
        qddot = planner.compute_action(**arguments)
        qdot[i+1, :] = qdot[i, :] + qddot * DT
        q[i+1, :] = q[i, :] + 0.5 * qdot[i, :] * DT + 0.5 * qdot[i+1, :] * DT
        if break_condition(q[i+1, :], qddot, goal):
            return q
    print(arguments['x_goal_0'])
    print(q[0, :])
    return q

def add_obstacle_circulare(ax, obstacle):
    ax.add_patch(plt.Circle(
        obstacle.position()[0:2],
        radius=obstacle.radius(),
        linewidth=0.5,
        edgecolor=obstacle_edge_color,
        facecolor=obstacle_color
    ))

def add_obstacle_with_blur(ax, obstacle):
    outer_radius = obstacle.radius() - 0
    inner_radius = obstacle.radius() - 0.5
    rgb = plt.cm.Blues(300)
    rgba = np.array([rgb[0], rgb[1], rgb[2], 0.7])
    radii = np.linspace(outer_radius, inner_radius, 20)
    alphas = np.linspace(0.7, 0.01, 20)
    alphas[0] = 1.0
    for alpha, radius in zip(alphas, radii):
        rgba[3] = alpha
        ax.add_patch(plt.Circle(
            obstacle.position()[0:2],
            radius=radius,
            linewidth=1,
            edgecolor=rgba,
            facecolor='none',
            zorder=9,
        ))

def add_obstacle_box(ax, obstacle):
    box_obstacle_center = [
        obstacle.position()[0] - 0.5 * obstacle.size()[0],
        obstacle.position()[1] - 0.5 * obstacle.size()[1]
    ]
    ax.add_patch(plt.Rectangle(
        box_obstacle_center,
        width=obstacle.size()[0],
        height=obstacle.size()[1],
        linewidth=0.5,
        edgecolor=obstacle_edge_color,
        facecolor=obstacle_color
    ))

def add_text(ax, text, position, fontsize=5, rotation=0, fontweight='normal'):
    ax.text(
        position[0],
        position[1],
        text,
        color='white',
        fontsize=fontsize*4,
        fontfamily='Montserrat',
        fontweight=fontweight,
        ha='center',
        va='center',
        rotation=rotation,
    )


def insert_png(ax, image_path, position, size):
    arr_img = plt.imread(image_path)
    # no background
    ax.set_facecolor('black')
    im = OffsetImage(arr_img, zoom=size[0])
    ab = AnnotationBbox(
        im,
        position,
        xybox=(0, 0),
        xycoords='data',
        boxcoords="offset points",
        bboxprops=dict(edgecolor='none', facecolor='none')
    )

    ax.add_artist(ab)

def plot_trajectories(file_path: str):
    trajectories = np.load(file_path, allow_pickle=True)
    not_plotting_indices = [32, 33, 34, 35, 40, 41, 61, 62, 63]
    not_plotting_indices = []
    for i, index in enumerate(not_plotting_indices):
        trajectories = np.delete(trajectories, index-i, axis=0)

    fig, ax = plt.subplots(figsize=(10 ,14), facecolor='black')
    ax.set_facecolor('black')
    N = len(trajectories)
    blue_colors = plt.cm.Blues(np.linspace(0.2, 1, N))

    for i, trajectory in enumerate(trajectories):
        q = trajectory[np.linalg.norm(trajectory, axis=1) > 1e-5]
        ax.plot(q[:, 0], q[:, 1], color=blue_colors[i], linewidth=1.5)
    plt.xlim(LIMITS["low"][0] - 0.1, LIMITS["high"][0] + 0.1)
    plt.ylim(LIMITS["low"][1]- 0.1, LIMITS["high"][1] + 0.1)
    # plot box for limits
    """
    ax.plot([LIMITS["low"][0], LIMITS["low"][0]], [LIMITS["low"][1], LIMITS["high"][1]], 'k', color='white')
    ax.plot([LIMITS["high"][0], LIMITS["high"][0]], [LIMITS["low"][1], LIMITS["high"][1]], 'k', color='white')
    ax.plot([LIMITS["low"][0], LIMITS["high"][0]], [LIMITS["low"][1], LIMITS["low"][1]], 'k', color='white')
    ax.plot([LIMITS["low"][0], LIMITS["high"][0]], [LIMITS["high"][1], LIMITS["high"][1]], 'k', color='white')
    """

    add_obstacle_box(ax, obst_0)
    add_obstacle_box(ax, obst_1)
    add_obstacle_box(ax, obst_2)
    add_obstacle_box(ax, obst_3)
    #add_obstacle_circulare(ax, obst_4)



    add_text(ax, "Invitation to the public defense\nof my PhD thesis", obst_0.position()[0:2], fontsize=4)

    title = "Trajectory Generation for Mobile\nManipulators with Differential Geometry"
    title_position = [obst_1.position()[0], obst_1.position()[1] + 1.0]
    add_text(ax, title, title_position, fontweight='bold', fontsize=4.7)

    info = "11 December 2024\nDefense: 17:30\nLayperson's talk: 17:00\nSenaatszaal\nAula Conference Center\nTU Delft"
    info_position = obst_2.position()
    add_text(ax, info, info_position, fontsize=3.5)

    name = "Max Spahn"
    email = "m.spahn@tudelft.nl"
    name_position = [obst_3.position()[0], obst_3.position()[1] + 0.85]
    email_position = [obst_3.position()[0], obst_3.position()[1] + 0.45]
    add_text(ax, name, name_position)
    add_text(ax, email, email_position, fontsize=4)


    for goal in user_defined_goal_positions:
        # color metalic orange
        rgb = np.array([234, 162, 33])/255
        # 9 radius starting with 0.2 to 0.4
        radii = np.linspace(0.15, 0.35, 20)
        alphas = np.linspace(0.08, 0.01, 20)
        alphas[0] = 1.0
        for alpha, radius in zip(alphas, radii):
            ax.add_patch(plt.Circle(
                goal,
                radius=radius,
                linewidth=2,
                edgecolor="none",
                facecolor=rgb,
                alpha=alpha,
                zorder=1000,
            ))
    for start in user_defined_initial_positions:
        # color blue
        rgb = np.array([153, 255, 255])/255
        radii = np.linspace(0.15, 0.35, 20)
        alphas = np.linspace(0.08, 0.01, 20)
        alphas[0] = 1.0
        for alpha, radius in zip(alphas, radii):
            ax.add_patch(plt.Circle(
                start,
                radius=radius,
                linewidth=2,
                edgecolor="none",
                facecolor=rgb,
                alpha=alpha,
                zorder=1000,
            ))
    ax.axis('equal')

    # save as vector graphics
    plt.savefig("invite_spahn.svg", format='svg', dpi=1200, bbox_inches='tight')
    plt.savefig("invite_spahn.png", format='png', dpi=100, bbox_inches='tight')

    #plt.show()

def generate_trajectories(n_trajectories: int = 10, n_steps: int = 1000, render: bool = False, filename: str = "point_robot_trajectories.npy"):
    planner = set_planner(default_goal)
    trajectories = []
    vel_mag = 3
    n_start = 15
    n_goal = 15
    initial_positions = []
    initial_velocities = []
    initial_position = np.zeros(2)
    for i in range(n_trajectories):
        print(i, n_start)
        if i % n_start == 0:
            print("new_start")
            initial_position = user_defined_initial_positions[i//n_start]
        initial_position = initial_position
        initial_velocity = np.random.uniform(-1, 1, 2)
        direction_angle = np.pi * 2 * (i % n_start)/n_start
        initial_velocity = vel_mag * np.array([
            np.cos(direction_angle),
            np.sin(direction_angle)
        ])
        #initial_velocity = vel_mag * np.array([0.5, -1])
        initial_positions.append(initial_position)
        initial_velocities.append(initial_velocity)
    goal = compose_goal(user_defined_goal_positions[0])
    if not render:
        setups = {}
        for i in range(n_trajectories):
            print(i)
            setups[i] = {
                "initial_position": initial_positions[i].tolist(),
                "initial_velocity": initial_velocities[i].tolist(),
                "goal": goal.primary_goal().position().tolist(),
            }


            if i % n_goal == 0:
                goal = compose_goal(user_defined_goal_positions[i//n_goal])
            trajectory = run_point_robot_rollout(
                planner,
                n_steps=n_steps,
                render=render,
                initial_position=initial_positions[i],
                initial_velocity=initial_velocities[i],
                goal=goal,
            )
            trajectories.append(trajectory)
    else:
        for i in range(n_trajectories):
            initial_position = initial_positions[i]
            initial_velocity = initial_velocities[i]
            trajectory = run_point_robot(
                planner,
                n_steps=n_steps,
                render=render,
                initial_position=initial_position,
                initial_velocity=initial_velocity,
                goal=goal,
            )
            trajectories.append(trajectory)
    # save trajectories 
    with open('setups_invite.yaml', 'w') as file:
        yaml.dump(setups, file)
    np.save(filename, trajectories)



if __name__ == "__main__":
    render = False
    n = 1
    render = bool(int(sys.argv[1]))
    n = int(sys.argv[2])
    name = sys.argv[3]
    #generate_trajectories(n_trajectories=n, n_steps=10000, render=render, filename=name)
    #run_point_robot(render=True)
    plot_trajectories(name)



