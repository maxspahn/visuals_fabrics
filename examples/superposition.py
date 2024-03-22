# pylint: disable=import-outside-toplevel
import sys
import numpy as np
import yaml
from visuals_fabrics.plot_trajectories import plot_trajectories
from visuals_fabrics.runners import run_point_robot, run_point_robot_rollout, compose_goal

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from forwardkinematics.planarFks.pointFk import PointFk

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# This example showcases the psedo-sensor and requires goals and obstacles.
# As a result, it requires the optional package motion_planning_scenes.

# Consider installing it with `poetry install -E scenes`.


obst_dict = {
    "type": "sphere",
    "geometry": {"position": [2.0, 0.0, 0.0], "radius": 1.0},
}
obst_0 = SphereObstacle(name="simpleSphere", content_dict=obst_dict)
obst_dict = {
    "type": "sphere",
    "geometry": {"position": [20000.0, 20000.0, 0.0], "radius": 1.0},
}
obst_0_inf = SphereObstacle(name="simpleSphere", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [0.5, 2, 0.0],
        "length": 2.3,
        "width": 1.0,
    },
}
obst_1 = BoxObstacle(name="simpleBox", content_dict=obst_dict)
obst_dict = {
    "type": "box",
    "geometry": {
        "position": [20000.5, 200000, 0.0],
        "length": 2.3,
        "width": 1.0,
    },
}
obst_1_inf = BoxObstacle(name="simpleBox", content_dict=obst_dict)

def set_planner(goal: GoalComposition, config: dict):
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
    """
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
        limits=limits_as_array,
    )
    """
    planner.load_fabrics_configuration(config['fabrics'])
    planner.load_problem_configuration(config['problem'])
    planner.concretize()
    return planner


def generate_trajectories(n_steps: int = 1000, render: bool = False):
    config_file = 'superposition_config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    limits = config['problem']['joint_limits']
    goal = compose_goal(config['problem'], np.array([3, 3]))
    obstacle_configs = [[obst_0, obst_1_inf], [obst_0_inf, obst_1], [obst_0, obst_1]]
    planner = set_planner(goal, config)
    initial_position = np.array([-3, -2])
    initial_velocities = []
    for j in range(10):
        angle = j * np.pi / 11
        initial_velocities.append(np.array([np.cos(angle), np.sin(angle)]))
    for i, obstacles in enumerate(obstacle_configs):
        trajectories = []
        arguments = get_arguments(goal, obstacles)
        for initial_velocity in initial_velocities:
            if not render:
                trajectory = run_point_robot_rollout(
                    planner,
                    arguments,
                    goal,
                    limits,
                    obstacles,
                    n_steps=n_steps,
                    render=render,
                    initial_position=initial_position,
                    initial_velocity=initial_velocity,
                )
                trajectories.append(trajectory)
            else:
                trajectory = run_point_robot(
                    planner,
                    arguments,
                    goal,
                    limits,
                    obstacles,
                    n_steps=n_steps,
                    render=render,
                    initial_position=initial_position,
                    initial_velocity=initial_velocity,
                )
                trajectories.append(trajectory)
        # save trajectories 
        np.save(f"superposition_{i}.npy", trajectories)
    return limits, obstacle_configs

def get_arguments(goal, obstacles):
    arguments = {
        'x_goal_0': goal.primary_goal().position(),
        'weight_goal_0': goal.primary_goal().weight(),
        'radius_body_1': 0.1,
        'x_obst_0': obstacles[0].position(),
        'radius_obst_0': obstacles[0].radius(),
        'x_obst_1': obstacles[1].position(),
        'sizes_obst_1': obstacles[1].size(),
    }
    return arguments



if __name__ == "__main__":
    render = False
    n = 1
    if len(sys.argv) > 1:
        render = bool(int(sys.argv[1]))
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    limits, obstacle_configs = generate_trajectories(n_steps=10000, render=render)
    #run_point_robot(render=True)
    for i, obstacles in enumerate(obstacle_configs):
        plot_trajectories(f"superposition_{i}.npy", limits, obstacles)



