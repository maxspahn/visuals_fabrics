# pylint: disable=import-outside-toplevel
from copy import deepcopy
import sys
from typing import List
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
import numpy as np
import yaml

import matplotlib.pyplot as plt

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from forwardkinematics.planarFks.pointFk import PointFk
from planarenvs.sensors.full_sensor import FullSensor

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# This example showcases the psedo-sensor and requires goals and obstacles.
# As a result, it requires the optional package motion_planning_scenes.

# Consider installing it with `poetry install -E scenes`.

CONFIG_FILE = "cover_config.yaml"
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)
    CONFIG_PROBLEM = config['problem']
    CONFIG_FABRICS = config['fabrics']

DT = 0.01

default_goal = GoalComposition(name="goal", content_dict=CONFIG_PROBLEM["goal"]['goal_definition'])
obst_dict = {
    "type": "sphere",
    "geometry": {"position": [-11.0, -6.0, 0.0], "radius": 6.0},
}
obst_0 = SphereObstacle(name="simpleSphere", content_dict=obst_dict)
obst_dict = {
    "type": "sphere",
    "geometry": {"position": [3.5, -5.0, 0.0], "radius": 3.0},
}
obst_1 = SphereObstacle(name="simpleSphere", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [7.5, -7, 0.0],
        "length": 8.0,
        "width": 2.0,
    },
}
obst_2 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [7.5, 7, 0.0],
        "length": 11,
        "width": 3,
    },
}
obst_3 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
LIMITS = {
    "high": np.array(CONFIG_PROBLEM['joint_limits']['upper_limits']),
    "low": np.array(CONFIG_PROBLEM['joint_limits']['lower_limits']),
}

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


def time_variant_action(t):
    return np.array([np.cos(t), np.sin(t)])


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
        if terminated or break_condition(q[i, :], action, goal):
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
            np.linalg.norm(goal_location - obst_0.position()[0:2]) < goal.primary_goal().epsilon() + obst_0.radius() or
            np.linalg.norm(goal_location - obst_1.position()[0:2]) < goal.primary_goal().epsilon() + obst_1.radius() or
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
        'x_obst_0': obst_0.position(),
        'radius_obst_0': obst_0.radius(),
        'x_obst_1': obst_1.position(),
        'radius_obst_1': obst_1.radius(),
        'x_obst_2': obst_2.position(),
        'sizes_obst_2': obst_2.size(),
        'x_obst_3': obst_3.position(),
        'sizes_obst_3': obst_3.size(),
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

def plot_trajectories(file_path: str):
    trajectories = np.load(file_path, allow_pickle=True)
    fig, ax = plt.subplots(facecolor='black')
    ax.set_facecolor('black')
    N = len(trajectories)
    blue_colors = plt.cm.Blues(np.linspace(0.2, 1, N))

    for i, trajectory in enumerate(trajectories):
        q = trajectory[np.linalg.norm(trajectory, axis=1) > 1e-5]
        ax.plot(q[:, 0], q[:, 1], color=blue_colors[i])
    plt.xlim(LIMITS["low"][0] - 0.1, LIMITS["high"][0] + 0.1)
    plt.ylim(LIMITS["low"][1]- 0.1, LIMITS["high"][1] + 0.1)
    # plot box for limits
    ax.plot([LIMITS["low"][0], LIMITS["low"][0]], [LIMITS["low"][1], LIMITS["high"][1]], 'k', color='white')
    ax.plot([LIMITS["high"][0], LIMITS["high"][0]], [LIMITS["low"][1], LIMITS["high"][1]], 'k', color='white')
    ax.plot([LIMITS["low"][0], LIMITS["high"][0]], [LIMITS["low"][1], LIMITS["low"][1]], 'k', color='white')
    ax.plot([LIMITS["low"][0], LIMITS["high"][0]], [LIMITS["high"][1], LIMITS["high"][1]], 'k', color='white')
    
    ax.add_patch(plt.Circle(
        obst_0.position(),
        radius=obst_0.radius(),
        linewidth=2,
        edgecolor='none',
        facecolor='white'
    ))
    ax.add_patch(plt.Circle(
        obst_1.position(),
        radius=obst_1.radius(),
        linewidth=2,
        edgecolor='none',
        facecolor='white'
    ))
    # plot rectangle patches
    box_obstacle_center = [
        obst_2.position()[0] - 0.5 * obst_2.size()[0],
        obst_2.position()[1] - 0.5 * obst_2.size()[1]
    ]
    ax.add_patch(plt.Rectangle(
        box_obstacle_center,
        width=obst_2.size()[0],
        height=obst_2.size()[1],
        linewidth=2,
        edgecolor='none',
        facecolor='white'
    ))
    box_obstacle_center = [
        obst_3.position()[0] - 0.5 * obst_3.size()[0],
        obst_3.position()[1] - 0.5 * obst_3.size()[1]
    ]
    ax.add_patch(plt.Rectangle(
        box_obstacle_center,
        width=obst_3.size()[0],
        height=obst_3.size()[1],
        linewidth=2,
        edgecolor='none',
        facecolor='white'
    ))
    ax.axis('equal')
    # save as vector graphics
    plt.savefig("point_robot_trajectories.svg")
    plt.show()

def generate_trajectories(n_trajectories: int = 10, n_steps: int = 1000, render: bool = False):
    planner = set_planner(default_goal)
    trajectories = []
    vel_mag = 3
    n_start = 15
    n_goal = 10
    initial_positions = []
    initial_velocities = []
    initial_position = np.zeros(2)
    user_defined_initial_positions = [
        np.array([0, 3]),
        np.array([12, 3]),
        np.array([9, -1]),
        np.array([-5, 0]),
        np.array([-6, 8]),
    ]
    user_defined_goal_positions = [
        np.array([-13, 8]),
        np.array([-2, -9]),
        np.array([8, -1]),
        np.array([-14, -1]),
        np.array([-4, 8]),
        np.array([0, 0]),
        np.array([0, -9]),
        np.array([-14, 0]),
        np.array([14, -9]),
        np.array([14, 7]),
    ]
    for i in range(n_trajectories):
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
        for i in range(n_trajectories):
            print(i)
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
    np.save("point_robot_trajectories.npy", trajectories)



if __name__ == "__main__":
    render = False
    n = 1
    if len(sys.argv) > 1:
        render = bool(int(sys.argv[1]))
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    #generate_trajectories(n_trajectories=n, n_steps=10000, render=render)
    #run_point_robot(render=True)
    plot_trajectories("point_robot_trajectories.npy")



