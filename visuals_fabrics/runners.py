from copy import deepcopy
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
import numpy as np

from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

DT = 0.01

def run_point_robot(
    planner: ParameterizedFabricPlanner,
    arguments: dict,
    goal: GoalComposition,
    limits: dict,
    obstacles: list,
    n_steps: int = 1000,
    render: bool = False,
    initial_position: np.ndarray = np.zeros(2),
    initial_velocity: np.ndarray = np.zeros(2),
):
    env = PointRobotAccEnv(render=render, dt=DT)
    limits_env = {
        "high": np.array(limits['upper_limits']),
        "low": np.array(limits['lower_limits']),
    }
    env.reset_limits(pos=limits_env)
    ob = env.reset(pos=initial_position, vel=initial_velocity)


    for obstacle in obstacles:
        env.add_obstacle(obstacle)
    env.add_goal(goal.sub_goals()[0])
    ob, *_ = env.step(np.zeros(2))

    print("Starting episode")
    q = np.zeros((n_steps, 2))
    qdot = np.zeros((n_steps, 2))


    for i in range(n_steps):
        q[i, :] = ob['joint_state']['position']
        qdot[i, :] = ob['joint_state']['velocity']
        arguments['q'] = ob['joint_state']['position']
        arguments['qdot'] = ob['joint_state']['velocity']
        action = planner.compute_action(**arguments)
        ob, reward, terminated, info = env.step(action)
        if terminated or break_condition(q[i, :], action, goal, limits):
            return q
    print(arguments['x_goal_0'])
    print(q[0, :])
    return q

def break_condition(q, action, goal, limits):
    outside_limits = q[0] > limits["upper_limits"][0] or q[0] < limits["lower_limits"][0] or q[1] > limits["upper_limits"][1] or q[1] < limits["lower_limits"][1]
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
            np.linalg.norm(goal_location - obst_0.position()[0:2]) <  obst_0.radius() or
            np.linalg.norm(goal_location - obst_1.position()[0:2]) <  obst_1.radius() or
            np.linalg.norm(goal_location - obst_2.position()[0:2]) <  obst_2.size()[0] or
            np.linalg.norm(goal_location - obst_3.position()[0:2]) <  obst_3.size()[0]
    )

def compose_goal(config_problem, goal_location):
    goal_dict = deepcopy(config_problem["goal"]['goal_definition'])
    goal_dict['subgoal0']['desired_position'] = goal_location.tolist()
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    return goal


def run_point_robot_rollout(
    planner: ParameterizedFabricPlanner,
    arguments: dict,
    goal: GoalComposition,
    limits: dict,
    obstacles: list,
    n_steps: int = 1000,
    render: bool = False,
    initial_position: np.ndarray = np.zeros(2),
    initial_velocity: np.ndarray = np.zeros(2),
):
    q = np.zeros((n_steps, 2))
    qdot = np.zeros((n_steps, 2))
    q[0, :] = initial_position
    qdot[0, :] = initial_velocity

    for i in range(n_steps-1):
        arguments['q'] = q[i, :]
        arguments['qdot'] = qdot[i, :]
        qddot = planner.compute_action(**arguments)
        qdot[i+1, :] = qdot[i, :] + qddot * DT
        q[i+1, :] = q[i, :] + 0.5 * qdot[i, :] * DT + 0.5 * qdot[i+1, :] * DT
        if break_condition(q[i+1, :], qddot, goal, limits):
            return q
    print(arguments['x_goal_0'])
    print(q[0, :])
    return q
