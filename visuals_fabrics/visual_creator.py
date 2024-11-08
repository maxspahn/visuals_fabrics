# pylint: disable=import-outside-toplevel
from copy import deepcopy
from typing import List
import cairosvg
import sys
import os
from typing import List
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
import numpy as np
import yaml

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from forwardkinematics.planarFks.pointFk import PointFk
from planarenvs.sensors.full_sensor import FullSensor

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner



class VisualCreator():
    _ax: plt.Axes
    _obstacles: List[CollisionObstacle] = []
    dt: float = 0.01

    def __init__(self, debug: bool = False):
        self.obstacle_color = 'none'
        self.obstacle_edge_color = 'white'
        self.help_color = 'white'
        self.debug = debug


    def set_config_files(
            self,
            trajectory_file_name: str,
            planner_config_file_name: str,
            start_end_setup_file_name: str,
        ):
        self._planner_config_file_name = planner_config_file_name
        self._trajectory_file_name = trajectory_file_name
        self._start_end_setup_file_name = start_end_setup_file_name

    def init_figure(self, width_mm: float, height_mm: float):
        width_in = width_mm / 25.4
        width_pixels = width_in * 72
        height_in = height_mm / 25.4
        height_pixels = height_in * 72
        print(f"Dimension in pixels {width_pixels}x{height_pixels}")


        fig = plt.figure(figsize=(width_in, height_in))
        fig.patch.set_facecolor('black')
        self.ax = fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.ax.margins(0)
        self.ax.set_facecolor('black')
        self.x0 = -width_mm / 20
        self.x1 = width_mm / 20
        self.y0 = -height_mm / 20
        self.y1 = height_mm / 20

    def compose_goal(self, goal_location: list) -> GoalComposition:
        goal_dict = deepcopy(self._config_problem["goal"]['goal_definition'])
        goal_dict['subgoal0']['desired_position'] = goal_location
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        return goal

    def insert_png(self, image_path, position, size):
        arr_img = plt.imread(image_path)
        # no background
        im = OffsetImage(arr_img, zoom=size)
        ab = AnnotationBbox(
            im,
            position,
            xybox=(0, 0),
            xycoords='data',
            boxcoords="offset points",
            bboxprops=dict(edgecolor='none', facecolor='none')
        )

        self.ax.add_artist(ab)

    def add_obstacle_circulare(self, obstacle):
        self.ax.add_patch(plt.Circle(
            obstacle.position()[0:2],
            radius=obstacle.radius(),
            linewidth=0.5,
            edgecolor=self.obstacle_edge_color,
            facecolor=self.obstacle_color
        ))

    def add_obstacle_with_blur(self, obstacle):
        outer_radius = obstacle.radius() - 0
        inner_radius = obstacle.radius() - 0.5
        rgb = plt.cm.Blues(300)
        rgba = np.array([rgb[0], rgb[1], rgb[2], 0.7])
        radii = np.linspace(outer_radius, inner_radius, 20)
        alphas = np.linspace(0.7, 0.01, 20)
        alphas[0] = 1.0
        for alpha, radius in zip(alphas, radii):
            rgba[3] = alpha
            patch = plt.Circle(
                obstacle.position()[0:2],
                radius=radius,
                linewidth=1,
                edgecolor=rgba,
                facecolor='none',
                zorder=9,
            )
            self.ax.add_patch(patch)

    def add_obstacle_box(self, obstacle):
        box_obstacle_center = [
            obstacle.position()[0] - 0.5 * obstacle.size()[0],
            obstacle.position()[1] - 0.5 * obstacle.size()[1]
        ]
        self.ax.add_patch(plt.Rectangle(
            box_obstacle_center,
            width=obstacle.size()[0],
            height=obstacle.size()[1],
            linewidth=0.5,
            edgecolor=self.obstacle_edge_color,
            facecolor=self.obstacle_color
        ))

    def add_text(self, text, position, fontsize, fontweight, rotation=0):
        self.ax.text(
            position[0],
            position[1],
            text,
            color='white',
            fontsize=fontsize,
            fontfamily='Montserrat',
            fontweight=fontweight,
            ha='center',
            va='center',
            rotation=rotation,
        )


    def set_planner(self):
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
        self._planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            forward_kinematics,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler,
            limit_geometry=limit_geometry,
            limit_finsler=limit_finsler,
            damper_beta=damper_beta,
            damper_eta=damper_eta,
        )
        self._planner.load_fabrics_configuration(self._config_fabrics)
        self._planner.load_problem_configuration(self._config_problem)
        self._planner.concretize()

    @property
    def limits(self):
        return {
            "high": np.array(self._config_problem['joint_limits']['upper_limits']),
            "low": np.array(self._config_problem['joint_limits']['lower_limits']),
        }

    @property
    def dummy_goal(self):
        return GoalComposition(name="goal", content_dict=self._config_problem["goal"]['goal_definition'])


    def run_point_robot(
        self,
        planner: ParameterizedFabricPlanner,
        goal: GoalComposition,
        n_steps: int = 1000,
        render: bool = False,
        initial_position: np.ndarray = np.zeros(2),
        initial_velocity: np.ndarray = np.zeros(2),
    ):
        env = PointRobotAccEnv(render=render, dt=self.dt)
        env.reset_limits(pos=self.limits)
        ob = env.reset(pos=initial_position, vel=initial_velocity)


        for obstacle in self._obstacles:
            env.add_obstacle(obstacle)

        env.add_goal(goal.sub_goals()[0])
        full_sensor = FullSensor(
            ['position', 'weight'],
            ['position', 'size'],
            variance=0.0
        )
        env.add_sensor(full_sensor)
        ob, *_ = env.step(np.zeros(2))

        print("Starting episode")
        q = np.zeros((n_steps, 2))
        qdot = np.zeros((n_steps, 2))
        arguments = self.get_arguments(goal)


        for i in range(n_steps):
            q[i, :] = ob['joint_state']['position']
            qdot[i, :] = ob['joint_state']['velocity']
            arguments['q'] = ob['joint_state']['position']
            arguments['qdot'] = ob['joint_state']['velocity']
            action = planner.compute_action(**arguments)
            ob, reward, terminated, info = env.step(action)
            if terminated or self.break_condition(q[i, :], action, goal):
                return q
        return q

    def get_arguments(self, goal):
        arguments = {
            'x_goal_0': goal.primary_goal().position(),
            'weight_goal_0': goal.primary_goal().weight(),
        }
        for obstacle_index, obstacle in enumerate(self._obstacles):
            if isinstance(obstacle, SphereObstacle):
                arguments[f'x_obst_{obstacle_index}'] = obstacle.position()
                arguments[f'radius_obst_{obstacle_index}'] = obstacle.radius()
            elif isinstance(obstacle, BoxObstacle):
                arguments[f'x_obst_{obstacle_index}'] = obstacle.position()
                arguments[f'sizes_obst_{obstacle_index}'] = obstacle.size()

        return arguments

    def break_condition(self, q: np.ndarray, action: np.ndarray, goal: GoalComposition):
        outside_limits = q[0] > self.limits["high"][0] or q[0] < self.limits["low"][0] or q[1] > self.limits["high"][1] or q[1] < self.limits["low"][1]
        low_action_magnitude = np.linalg.norm(action) < 1e-4
        goal_reached = np.linalg.norm(q - goal.primary_goal().position()) < goal.primary_goal().epsilon()
        if low_action_magnitude:
            print("Low action magnitude")
        if outside_limits:
            print("Outside limits")
        if goal_reached:
            print("Goal reached")
        return outside_limits or low_action_magnitude or goal_reached




    def run_point_robot_rollout(
        self,
        planner: ParameterizedFabricPlanner,
        goal: GoalComposition,
        n_steps: int = 1000,
        render: bool = False,
        initial_position: np.ndarray = np.zeros(2),
        initial_velocity: np.ndarray = np.zeros(2),
    ):
        q = np.zeros((n_steps, 2))
        qdot = np.zeros((n_steps, 2))
        q[0, :] = initial_position
        qdot[0, :] = initial_velocity
        arguments = self.get_arguments(goal)

        for i in range(n_steps-1):
            arguments['q'] = q[i, :]
            arguments['qdot'] = qdot[i, :]
            qddot = planner.compute_action(**arguments)
            qdot[i+1, :] = qdot[i, :] + qddot * self.dt
            q[i+1, :] = q[i, :] + 0.5 * qdot[i, :] * self.dt + 0.5 * qdot[i+1, :] * self.dt
            if self.break_condition(q[i+1, :], qddot, goal):
                return q
        return q

    def load_configs(self):
        with open(self._planner_config_file_name, 'r') as config_file:
            config = yaml.safe_load(config_file)
            self._config_problem = config['problem']
            self._config_fabrics = config['fabrics']
            self._config_obstacles = config['obstacles']
            self._config_setup = config['setup']
            self._config_visuals = config['visuals']

        for obstacle_name, obstacle_config in self._config_obstacles.items():
            if obstacle_config['type'] == 'sphere':
                self._obstacles.append(SphereObstacle(
                    name=obstacle_name,
                    content_dict=obstacle_config
                ))
            elif obstacle_config['type'] == 'box':
                self._obstacles.append(BoxObstacle(
                    name=obstacle_name,
                    content_dict=obstacle_config
                ))
        self._goal_list = [np.array(goal) for goal in self._config_setup['goal_list']]
        self._start_list = [np.array(start) for start in self._config_setup['start_list']]
        self.vel_mag = self._config_setup['vel_mag']
        self.n_start = self._config_setup['n_start']
        self.n_goal = self._config_setup['n_goal']

    def load_setup_data(self):
        self._setup = yaml.safe_load(open(self._start_end_setup_file_name, 'r'))
        self._goal_positions = [np.array(setup['goal']) for setup in self._setup.values()]
        self._initial_positions = [np.array(setup['initial_position']) for setup in self._setup.values()]
        self._initial_velocities = [np.array(setup['initial_velocity']) for setup in self._setup.values()]



    def plot_trajectories(self):
        trajectories = np.load(self._trajectory_file_name, allow_pickle=True)
        N = len(trajectories)
        blue_colors = plt.cm.Blues(np.linspace(0.2, 1, N))
        blue_colors_rgb = np.array([blue_colors[:, 0], blue_colors[:, 1], blue_colors[:, 2]]).T * 255

        not_plotting_indices = []
        for i, trajectory in enumerate(trajectories):
            if i in not_plotting_indices:
                print("Ignoring index", i)
                continue
            q = trajectory[np.linalg.norm(trajectory, axis=1) > 1e-5]
            distance_to_all_goals = [np.linalg.norm(q[-1,:] - self._goal_positions[i], axis=0) for i in range(len(self._goal_positions))]

            if np.min(distance_to_all_goals) > 0.1:
                print("Not reached goal", i)
                continue
            self.ax.plot(q[:, 0], q[:, 1], color=blue_colors[i], linewidth=1.0)
            # plot index at at position q[0,:] + 0.1
            if self.debug:
                self.ax.text(q[100, 0], q[100, 1], str(i), color='white', fontsize=7.0)
        plt.xlim(self.limits["low"][0] - 0.1, self.limits["high"][0] + 0.1)
        plt.ylim(self.limits["low"][1]- 0.1, self.limits["high"][1] + 0.1)
        # plot box for limits
        if self.debug:
            diagonal_color = 'white'
        else:
            diagonal_color = 'none'
        self.ax.plot([self.x0, self.x1], [self.y0, self.y1], color=diagonal_color)
        if self.debug:
            self.ax.plot([self.limits["low"][0], self.limits["low"][0]], [self.limits["low"][1], self.limits["high"][1]], color='white')
            self.ax.plot([self.limits["high"][0], self.limits["high"][0]], [self.limits["low"][1], self.limits["high"][1]], color='white')
            self.ax.plot([self.limits["low"][0], self.limits["high"][0]], [self.limits["low"][1], self.limits["low"][1]], color='white')
            self.ax.plot([self.limits["low"][0], self.limits["high"][0]], [self.limits["high"][1], self.limits["high"][1]], color='white')


        if self.debug:
            for obstacle in self._obstacles:
                if isinstance(obstacle, SphereObstacle):
                    self.add_obstacle_circulare(obstacle)
                elif isinstance(obstacle, BoxObstacle):
                    self.add_obstacle_box(obstacle)
                self.add_text(obstacle.name(), obstacle.position()[0:2], 8.0, 'normal')


        for goal_index, goal in enumerate(self._goal_list):
            # color metalic orange
            rgb = np.array([234, 162, 33])/255
            if self.debug:
                self.ax.text(goal[0]+0.5, goal[1], str(goal_index), color=rgb, fontsize=7.0)
            # 9 radius starting with 0.2 to 0.4
            radii = np.linspace(0.15, 0.35, 20)
            alphas = np.linspace(0.08, 0.01, 20)
            alphas[0] = 1.0
            for alpha, radius in zip(alphas, radii):
                self.ax.add_patch(plt.Circle(
                    goal,
                    radius=radius,
                    linewidth=2,
                    edgecolor="none",
                    facecolor=rgb,
                    alpha=alpha,
                    zorder=1000,
                ))
        for start_index, start in enumerate(self._start_list):
            rgb = np.array([153, 255, 255])/255
            if self.debug:
                self.ax.text(start[0]+0.5, start[1], str(start_index), color=rgb, fontsize=7.0)
            radii = np.linspace(0.15, 0.35, 20)
            alphas = np.linspace(0.08, 0.01, 20)
            alphas[0] = 1.0
            for alpha, radius in zip(alphas, radii):
                self.ax.add_patch(plt.Circle(
                    start,
                    radius=radius,
                    linewidth=2,
                    edgecolor="none",
                    facecolor=rgb,
                    alpha=alpha,
                    zorder=1000,
                ))
        self.ax.axis('equal')

    def get_obstacle_by_name(self, name) -> CollisionObstacle:
        for obstacle in self._obstacles:
            if obstacle.name() == name:
                return obstacle
        raise ValueError(f"Obstacle with name {name} not found")

    def plot_visuals(self):
        for visual_name, visual in self._config_visuals.items():
            if 'text' in visual:
                text = visual['text']
                obstacle = self.get_obstacle_by_name(visual['position']['obstacle_name'])
                visual_position = obstacle.position()[0:2] + np.array(visual['position']['offset'])
                fontweight = visual['fontweight'] if 'fontweight' in visual else 'bold'
                rotation = visual['rotation'] if 'rotation' in visual else 0
                self.add_text(text, visual_position, visual['fontsize'], fontweight, rotation=rotation)
            elif 'img' in visual:
                image_path = visual['img']
                obstacle = self.get_obstacle_by_name(visual['position']['obstacle_name'])
                visual_position = obstacle.position()[0:2] + np.array(visual['position']['offset'])
                self.insert_png(image_path, visual_position, visual['scale'])

    def create_visual(self, output_file_name: str):
        self.plot_trajectories()
        self.plot_visuals()
        # save as vector graphics
        plt.savefig(f"{output_file_name}.svg", format='svg', dpi=1200)
        # convert to pdf
        cairosvg.svg2pdf(url=f"{output_file_name}.svg", write_to=f"{output_file_name}.pdf")


    @property
    def setup_file_exists(self):
        return os.path.exists(self._start_end_setup_file_name)

    def generate_trajectories(self, n_trajectories: int = 10, n_steps: int = 10000, render: bool = False):
        planner = self.set_planner()
        trajectories = []
        n_trajectories = min(self.n_start * len(self._start_list), self.n_goal * len(self._goal_list))
        print(f"Running {n_trajectories} trajectories")

        self._setup = {}
        if self.setup_file_exists:
            self.load_setup_data()
        else:
            initial_positions = []
            initial_velocities = []
            initial_position = np.zeros(2)
            for i in range(n_trajectories):
                print(i, self.n_start)
                if i % self.n_start == 0:
                    initial_position = self._start_list[i//self.n_start]
                initial_position = initial_position
                initial_velocity = np.random.uniform(-1, 1, 2)
                direction_angle = np.pi * 2 * (i % self.n_start)/self.n_start
                initial_velocity = self.vel_mag * np.array([
                    np.cos(direction_angle),
                    np.sin(direction_angle)
                ])
                #initial_velocity = vel_mag * np.array([0.5, -1])
                initial_positions.append(initial_position)
                initial_velocities.append(initial_velocity)
            for i in range(n_trajectories):
                self._setup[i] = {
                    "initial_position": initial_positions[i].tolist(),
                    "initial_velocity": initial_velocities[i].tolist(),
                    "goal": self._goal_list[i//self.n_goal].tolist(),
                }
        if not render:
            for i in range(n_trajectories):
                goal = self.compose_goal(self._setup[i]['goal'])
                trajectory = self.run_point_robot_rollout(
                    self._planner,
                    goal,
                    n_steps=n_steps,
                    render=render,
                    initial_position=self._setup[i]['initial_position'],
                    initial_velocity=self._setup[i]['initial_velocity'],
                )
                trajectories.append(trajectory)
        else:
            for i in range(n_trajectories):
                goal = self.compose_goal(self._setup[i]['goal'])
                trajectory = self.run_point_robot(
                    self._planner,
                    goal,
                    n_steps=n_steps,
                    render=render,
                    initial_position=self._setup[i]['initial_position'],
                    initial_velocity=self._setup[i]['initial_velocity'],
                )
                trajectories.append(trajectory)
        # save trajectories 
        with open(self._start_end_setup_file_name, 'w') as file:
            yaml.dump(self._setup, file)
        np.save(self._trajectory_file_name, trajectories)







