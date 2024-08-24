from shutil import move
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from mlpro.rl.models import *
from abb_rl.env.mlpro_abb_sim_fhswf import ABBRoboterFhswfEnv
from abb_python_controller.controller import ABBController
from abb_rl.utils.geo_operations import euler_angles_to_matrix, quat_to_rot_matrix
from pathlib import Path
import torch
from scipy.spatial import distance
import signal
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from mlpro.wrappers.sb3 import *
from abb_rl.trainer.kinenn import *
import optuna


def handler(signum, frame):
    exit(1)

signal.signal(signal.SIGINT, handler)

# class CustomPPO(PPO):
#         def __init__(self, policy, env, **kwargs):
#             super().__init__(policy=policy, env=env, **kwargs)
#             kwargs['ortho_init'] = False
#             print("Print setup started...................................................")

# class CustomActorCriticPolicy(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Callable[[float], float],
#         *args,
#         **kwargs,
#     ):
#         # Disable orthogonal initialization
#         kwargs["ortho_init"] = False
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )

# 1 Update the environmnet
class MyABBRoboterFhswfEnv_geck_V1_2(ABBRoboterFhswfEnv):
    C_NAME = "ABB Roboter FHSWF Env"
    C_CYCLE_LIMIT = 500

    # def setup_spaces(self):
    #     self.action_counter=0

    def _set_additional_space(self, state_space, action_space):
        # Target Position
        state_space.add_dim(Dimension('Tx', Dimension.C_BASE_SET_R, 'Target X', '', 'point',
                                    '', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('Ty', Dimension.C_BASE_SET_R, 'Target Y', '', 'point',
                                    '', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('Tz', Dimension.C_BASE_SET_R, 'Target Z', '', 'point',
                                    '', [-np.inf, np.inf]))

        # Target Orientation
        state_space.add_dim(Dimension('TRoll', Dimension.C_BASE_SET_R, 'Target Roll', '', 'rad',
                                    '', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('TPitch', Dimension.C_BASE_SET_R, 'Target Pitch', '', 'rad',
                                    '', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('TYaw', Dimension.C_BASE_SET_R, 'Target Yaw', '', 'rad',
                                    '', [-np.inf, np.inf]))

        # Collision
        state_space.add_dim(Dimension('Coll', Dimension.C_BASE_SET_Z, 'Collision', '', 'coll',
                                    '', [0,1]))

        # EEF Distance Error (Current Position to Target Position)
        state_space.add_dim(Dimension('Delta_Tx', Dimension.C_BASE_SET_R, 'Distance Error to Target X', '', 'point',
                                    '', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('Delta_Ty', Dimension.C_BASE_SET_R, 'Distance Error to Target Y', '', 'point',
                                    '', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('Delta_Tz', Dimension.C_BASE_SET_R, 'Distance Error to Target Z', '', 'point',
                                    '', [-np.inf, np.inf]))

        # EEF Orientation Error
        state_space.add_dim(Dimension('Delta_TO', Dimension.C_BASE_SET_R, 'Orientation Error', '', 'point',
                                    '', [-np.inf, np.inf]))
              
        return state_space, action_space

    def _get_additional_states(self):
        return [*self._current_target, self._get_collision(), *self._get_distance_error()]

    def _reset_target(self):
        self._reset_collision()
        self._despawn_object()
        self._spawn_object()
        if self._current_target is None:
            self._current_target = self._get_target()
        self.old_pos = None

    def _define_target(self):
        self._current_target = None

    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        state_new = p_state_new.get_values()
        current_pos = state_new[6:9]
        current_orientation = state_new[9:13]
        target_pos = state_new[13:16]
        target_orientation = state_new[16:19]
        print(current_pos,target_pos)

        # Reward Function here
        if self._compute_broken (p_state_new) == True:
            total_reward = -10**4
        elif self._compute_success(p_state_new) == True:
            total_reward = 10**4
        # elif distance.euclidean(current_pos,target_pos) < 0.5000:
        #     total_reward = 1/(distance.euclidean(current_pos,target_pos)**3)
        # elif state_new[23] < 0.2000:
        #     total_reward = 50
        else:
            total_reward = 1/(distance.euclidean(current_pos,target_pos)**3)
        reward = Reward(Reward.C_TYPE_OVERALL)

        # Set Reward
        reward.set_overall_reward(total_reward)
        return reward

    def _compute_broken(self, p_state: State) -> bool:
        state_new = p_state.get_values()
        
        #if state_new[19:20][0]:
        if state_new[19]:
            self._state.set_terminal(True)
            return True
        elif state_new[6] <= 0.4:
            self._state.set_terminal(True)
        elif state_new[7] <= -0.2000 or state_new[7] >= 0.2000:

            self._state.set_terminal(True)
            return True
        elif state_new[8] <= 0.8000 or state_new[8] >= 1.4000:
            self._state.set_terminal(True)
            return True
        else:
            return False

    def _compute_success(self, p_state: State) -> bool:
        # This implementation only for the first task, which is pick the part from the belt
        state = p_state.get_values()

        # Orientation Error
        #Convert everything to rotation matrix
        '''target_rot_mat = euler_angles_to_matrix(torch.Tensor(state[16:19]), "XYZ")
        current_rot_mat = torch.squeeze(quat_to_rot_matrix(torch.Tensor([state[9:13]])))

        orient_error = 0
        # Calculate the total angle from the rotation matrix directly
        angle = acos(tr(R) - 1)/2
        target_angle = torch.acos((torch.trace(target_rot_mat) - 1)/2)
        current_angle = torch.acos((torch.trace(current_rot_mat) - 1)/2)        
        orient_error = -torch.abs(target_angle - current_angle).item()'''
        orient_error = state[23]

        #if distance.euclidean(state[6:9], state[13:16]) <= self.offset_to_target and orient_error >= -self.offset_to_target_orientation:
        if distance.euclidean(state[6:9], state[13:16]) <= self.offset_to_target: #and orient_error <= self.offset_to_target_orientation:
            self._state.set_terminal(True)
            return True
        else:
            return False

    def _reset_collision(self):
        self._robot_controller.set_io("DO_Collision", "0")

    def _spawn_object(self):
        self._robot_controller.set_io("DO_Create", "1")
        self._robot_controller.set_io("DO_Create", "0")

    def _despawn_object(self):
        self._robot_controller.set_io("DO_Sink", "1")
        self._robot_controller.set_io("DO_Sink", "0")

    def _gripper_on(self):
        self._robot_controller.set_io("DO_Grip", "1")

    def _gripper_off(self):
        self._robot_controller.set_io("DO_Grip", "0")

    def _get_collision(self):
        # return int(self._robot_controller.get_io("DO_Collision"))
        # return int(not self._motion_plan)
        return self._collision

    def _get_target(self):
        # Position
        val_posx = float(self._robot_controller.get_io("AI_PosX"))
        val_posy = float(self._robot_controller.get_io("AI_PosY"))
        val_posz = float(self._robot_controller.get_io("AI_PosZ"))

        # Orientation
        val_rotx = float(self._robot_controller.get_io("AI_RotX"))
        val_roty = float(self._robot_controller.get_io("AI_RotY"))
        val_rotz = float(self._robot_controller.get_io("AI_RotZ"))

        return [val_posx, val_posy, val_posz, val_rotx, val_roty, val_rotz]

    def _get_distance_error(self):
        # Current Position and Orientation
        ee_pose = self._robot_controller.get_ee_pose()

        # Posision Error
        pose_error_x = ee_pose[0]-self._current_target[0]
        pose_error_y = ee_pose[1]-self._current_target[1]
        pose_error_z = ee_pose[2]-self._current_target[2]
        
        # Target Orientationss
        target_orientation = [self._current_target[3], self._current_target[4], self._current_target[5]]
        current_orientation = [ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]]
        _target_rot_mat = euler_angles_to_matrix(torch.Tensor(target_orientation), "XYZ")
        _current_rot_mat = torch.squeeze(quat_to_rot_matrix(torch.Tensor(current_orientation)))

        # Calculate the total angle from the rotation matrix directly
        # angle = acos(tr(R) - 1)/2
        _target_angle = torch.acos((torch.trace(_target_rot_mat) - 1)/2)
        _current_angle = torch.acos((torch.trace(_current_rot_mat) - 1)/2)

        # Orientation Error
        _orient_error = torch.abs(_target_angle - _current_angle).item()

        if math.isnan(_orient_error):
            _orient_error = 0

        return [pose_error_x, pose_error_y, pose_error_z, _orient_error]

    def _reset(self, p_seed=None) -> None:
        self._robot_controller.home()
        self._robot_controller.set_io("DO_Collision", "0")
        self._robot_controller.reset()
        self._motion_plan = False
        self._reset_target()
        
        state = State(self._state_space)
        state.set_values(self._get_all_states())
        self._set_state(state)

        self._motion_plan = True

    def _extract_observation(self, p_state: State) -> State:
        if p_state.get_related_set() == self.get_state_space():
            return p_state

        obs_space = self.get_state_space()
        obs_dim_ids = obs_space.get_dim_ids()
        observation = State(obs_space)

        for dim_id in obs_dim_ids:
            p_state_ids = p_state.get_dim_ids()
            obs_idx = obs_space.get_dim_ids().index(dim_id)
            observation.set_value(dim_id, p_state.get_value(p_state_ids[obs_idx]))

        return observation    

    def _get_quaternion_from_euler(self, rpy):

        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
        return [qw,qx,qy,qz]
    
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        ik_model = KineNNInverse(6)
        
        self.action_counter += 1
        print(self.action_counter)
        if self.action_counter<0:
                obs = p_state.get_values()
                obs[1] = obs[1]-1.5708
                obs[5] = obs[5]-3.1416
                del obs[-5:]
                a = obs [-3:]
                del obs [-3:]
                quaternion = self._get_quaternion_from_euler(a)
                obs = obs + quaternion
                # print('zain',obs)
            # remove idx 18-21
            # obs to torch
            # input to kineNN
            
                obs = torch.Tensor(obs).reshape(1, len(obs))
                action = ik_model.forward(obs)# KineNN
                action = action.tolist()
                print(action)
                p_action_id = p_action.get_elem_ids()[0]
                p_action_elem = p_action.get_elem(p_action_id)
                for x in range(6):
                    p_action_elem_id = p_action_elem.get_dim_ids()[x]
                    p_action_elem.set_value(p_action_elem_id,action[x])
            

        return super()._simulate_reaction(p_state, p_action)
    


# 2 Implement your own RL scenario
class MyScenario(RLScenario):
    C_NAME = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 2.1 Setup environment
        robot_studio_ip = "192.168.186.1"

        p_dt = 0.05 # delta time for a time step
        p_offset_pose = 0.05 # offset to target pose
        p_offset_orient = 0.05 # offset to target orientation
        p_max_velocity = 0.15
        self.p_per_eps_cyclimit = 200#added externally
        p_cycle_limit = self.p_per_eps_cyclimit

        self._env = MyABBRoboterFhswfEnv_geck_V1_2(
                                    robot_ip=robot_studio_ip,
                                    robot_type=ABBController.ABB_IRB_1200_90,
                                    move_joint_with_moveit=False,
                                    dt=p_dt,
                                    p_action_mode=MyABBRoboterFhswfEnv_geck_V1_2.C_ACTION_JOINT_VELOCITY,
                                    p_offset=p_offset_pose,
                                    p_offset_orientation=p_offset_orient,
                                    p_max_velocity=p_max_velocity,
                                    p_cycle_limit=p_cycle_limit)

        # 2.2 Setup Policy
        # Random Actions (You need to apply an RL policy here!)
        ss_ids = self._env.get_state_space().get_dim_ids()

        # Joint Angles, EEF Pose Errors, EEF Orientation Errors, see attached notes for more information
        ss_ = [ss_ids[0], ss_ids[1], ss_ids[2], ss_ids[3], ss_ids[4], ss_ids[5], ss_ids[20], ss_ids[21], ss_ids[22], ss_ids[23]]
        ik_model=KineNNInverse(6)

        
        def objective(self,trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
            n_steps = trial.suggest_int('n_steps', 16, 2048)
            batch_size = trial.suggest_int('batch_size', 8,256)
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
            clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
            ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2)
    
            # Instantiate the PPO model with these hyperparameters
            model = PPO(
                policy='MlpPolicy',
                env=self._env,  # Ensure 'env' is defined earlier in your script
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=0
            )
        
            # Train the model
            model.learn(total_timesteps=2000)  # Adjust timesteps as needed
        
            # Evaluate the model
            mean_reward, _ = evaluate_policy(model, self._env, n_eval_episodes=10)
        
            return mean_reward
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial), n_trials=50)

        # Print out the best hyperparameters found and the corresponding reward
        print("Best hyperparameters: ", study.best_params)
        print("Best trial value (mean reward): ", study.best_value)
       

        #policy_random = RandomGenerator(p_observation_space=self._env.get_state_space().spawn(ss_), 
        #                        p_action_space=self._env.get_action_space(),
        #                        p_buffer_size=1,
        #                        p_ada=1,
        #                        p_logging=p_logging)   


        # PPO
        
        
        
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=5,
            env=None,
            _init_setup_model=False,
            device="cpu")
        
        # A2C
        # policy_sb3 = A2C(
        #             policy="MlpPolicy", 
        #             env=None,
        #             use_rms_prop=False, 
        #             _init_setup_model=False,
        #             device="cpu")
        
        # DQN Discrete only
        # policy_sb3 = DQN(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False,
        #             device="cpu")

        # DDPG Continuous only
        # policy_sb3 = DDPG(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False,
        #             device="cpu")

        # SAC Continuous only


        # policy_sb3 = SAC(
        #             policy="MlpPolicy", 
        #             env=None, 
        #             _init_setup_model=False,
        #             device="cpu")
        
        
       
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_kinneninverse=ik_model,
            p_per_ep_cycle = self.p_per_eps_cyclimit,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_logging=p_logging)

        # 2.3 Setup standard single-agent with the policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=None,
            p_name='ABBRobotArm',
            p_ada=p_ada,
            p_logging=p_logging
        )


# 3 Create scenario and start training

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = str(Path.cwd()) + "/rl_results"
else:
    # 3.2 Parameters for internal unit test
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = None

# 3.3 Create and run training object
p_min_eps = 100
p_per_ep_cycle = 300
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=p_min_eps*p_per_ep_cycle, #min_ep*cycle_limit; was just int values earlier
    p_max_adaptations=0,
    p_max_stagnations=0,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging)
sys.exit()
try:
    training.run()
except KeyboardInterrupt:
    exit()
