# pylint: disable=import-outside-toplevel
import gym
from planarenvs.point_robot.envs.vel import PointRobotVelEnv
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
import numpy as np

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from forwardkinematics.planarFks.pointFk import PointFk
from planarenvs.sensors.full_sensor import FullSensor

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# This example showcases the psedo-sensor and requires goals and obstacles.
# As a result, it requires the optional package motion_planning_scenes.

# Consider installing it with `poetry install -E scenes`.

DT = 0.01

goal_dict = {
    "subgoal0": {
        "weight": 8.0,
        "is_primary_goal": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 2,
        "desired_position": [8, 1],
        "epsilon": 0.02,
        "type": "staticSubGoal",
    },
}
goal = GoalComposition(name="goal", content_dict=goal_dict)
obst_dict = {
    "type": "sphere",
    "geometry": {"position": [0.5, 0.7], "radius": 0.2},
}
obst_1 = SphereObstacle(name="simpleSphere", content_dict=obst_dict)
LIMITS = {
    "high": np.array([2.0, 3.0]),
    "low": np.array([-2.0, -3.0]),
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
        "-0.1 / (x ** 1) * xdot ** 2"
    )
    limit_finsler: str = (
        "1.0/(x**2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
    )
    forward_kinematics = PointFk()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        limit_geometry=limit_geometry,
        limit_finsler=limit_finsler,
    )
    collision_links = [1]
    # The planner hides all the logic behind the function set_components.
    # convert LIMITS to 2d array
    limits_as_array = [
        [LIMITS["low"][0], LIMITS["high"][0]],
        [LIMITS["low"][1], LIMITS["high"][1]],
    ]
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
        limits=limits_as_array,
    )
    planner.concretize()
    return planner


def time_variant_action(t):
    return np.array([np.cos(t), np.sin(t)])


def run_point_robot(
    n_steps: int = 1000,
    render: bool = False,
):
    env = PointRobotAccEnv(render=render, dt=DT)
    init_pos = np.array([0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    ob = env.reset(pos=init_pos, vel=init_vel)
    env.reset_limits(pos=LIMITS)

    planner = set_planner(goal)

    env.add_obstacle(obst_1)
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
    for i in range(n_steps):
        action = time_variant_action(env.t())
        arguments = {
            'q' : ob['joint_state']['position'],
            'qdot' : ob['joint_state']['velocity'],
            'x_goal_0': ob['goals'][0][0],
            'weight_goal_0': ob['goals'][0][1],
            'radius_body_1': 0.1,
            'x_obst_0': ob['obstacles'][0][0],
            'radius_obst_0': ob['obstacles'][0][1],
        }
        action = planner.compute_action(**arguments)
        ob, _, _, _ = env.step(action)
        observation_history.append(ob)
        if i % 100 == 0:
            print(f"ob : {ob}")
    return observation_history


if __name__ == "__main__":
    run_point_robot(render=True)
