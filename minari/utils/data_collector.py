import gymnasium as gym
from gymnasium import spaces
import tempfile
import h5py
import shutil
import os
import numpy as np

STEP_DATA_KEYS = set(['actions', 'observations', 'rewards', 'truncations', 'terminations', 'infos'])

class EpisodeMetadataCallback(object):
    def __call__(self, eps_group):
        eps_group['rewards'].attrs['sum'] = np.sum(eps_group['rewards'])
        eps_group['rewards'].attrs['mean'] = np.mean(eps_group['rewards'])
        eps_group['rewards'].attrs['std'] = np.std(eps_group['rewards'])
        eps_group['rewards'].attrs['max'] = np.max(eps_group['rewards'])
        eps_group['rewards'].attrs['min'] = np.min(eps_group['rewards'])
        
        eps_group.attrs['total_steps'] = eps_group['actions'].shape[0]


class StepPreProcessor(object):
    def __init__(self, env):
        self.env = env

        def check_flatten_space(space):
            """Check if space needs to be flatten or if it's not supported by Minari.

            Args:
                space: the Gymnasium space to be checked

            Returns:
                bool: True if space needs to be flatten before storing in hdf5 dataset. False otherwise.
            
            ValueError: If space is/contains Text, Sequence, or Graph space types
            """
            if isinstance(space, (spaces.Dict, spaces.Tuple)):
                for s in space.spaces.values():
                    check_flatten_space(s)
                return True
            elif isinstance(self.env.observation_space, (spaces.Text, spaces.Sequence, spaces.Graph)):
                ValueError (f"Minari doesn't support space of type {space}")            
            else:
                return False
        
        self.flatten_observation = check_flatten_space(self.env.observation_space)
        self.flatten_action = check_flatten_space(self.env.action_space)
    
    def __call__(self, env, obs, info, action=None, rew=None, terminated=None, truncated=None):
        if action is not None:
            # Flatten the actions
            if self.flatten_action:
                action = spaces.utils.flatten(self.env.action_space, action)
        # Flatten the observations
        if self.flatten_observation:
            obs = spaces.utils.flatten(self.env.observation_space, obs)
            
        step_data = {'actions': action, 'observations': obs, 'rewards': rew, 'terminations': terminated,
                        'truncations': truncated, 'infos': info}
        
        return step_data


class DataCollectorV0(gym.Wrapper):
    """Gymnasium environment wrapper to log stepping data.

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env, step_preprocessor=StepPreProcessor, episode_metadata_callback=EpisodeMetadataCallback, record_infos=False, max_steps_buffer=None, max_episodes_buffer=None):
        self.env = env
        self._step_preprocessor = step_preprocessor(env)
        
        self._episode_metadata_callback = episode_metadata_callback()
        self._record_infos = record_infos
        
        if max_steps_buffer is not None and max_episodes_buffer is not None:
            raise ValueError("Choose step or episode scheduler not both")
        
        self.max_episodes_buffer = max_episodes_buffer
        self.max_steps_buffer = max_steps_buffer

        self._buffer = [{'observations': [], 'actions': [], 'terminations': [], 'truncations': [], 'rewards': []}]
        
        self._current_seed = None
        self._new_episode = False
        
        self._step_id = 0
        self.datasets_path = os.environ.get('MINARI_DATASETS_PATH')
        if self.datasets_path is None:
            self.datasets_path = os.path.join(os.path.expanduser('~'), '.minari', 'datasets')
            
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self.datasets_path)
        self._tmp_f = h5py.File(os.path.join(self._tmp_dir.name, "tmp_dataset.hdf5"), 'a', track_order=True) # track insertion order of groups ('episodes')
        self._tmp_f.attrs['env_spec'] = self.env.spec.to_json()
        self._tmp_f.attrs['flatten_observation'] = self._step_preprocessor.flatten_observation
        self._tmp_f.attrs['flatten_action'] = self._step_preprocessor.flatten_action
        
        self._episode_id = 0
        self._new_episode = False
        self._eps_group = None
        self._last_episode_group_term_or_trunc = True # Generate new episode  
        self._last_episode_n_steps = 0              
        
    def _add_to_buffer(self, buffer, step_data):
        for key, value in step_data.items():
            if not self._record_infos and key == 'infos':
                continue
            if key not in buffer:
                if isinstance(value, dict):
                    buffer[key] = self._add_to_buffer({}, value)      
                else:
                    buffer[key] = [value]
            else:
                if isinstance(value, dict):
                    buffer[key] = self._add_to_buffer(buffer[key], value)      
                else:
                    buffer[key].append(value)
        
        return buffer
                        
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        step_data = self._step_preprocessor(env=self, obs=obs, info=info, action=action, rew=rew, terminated=terminated, truncated=truncated)
        
        assert STEP_DATA_KEYS.issubset(step_data.keys())
         
        self._step_id += 1
        
        if self.max_steps_buffer is not None:
            clear_buffers = (self._step_id % self.max_steps_buffer == 0)
        else:
            clear_buffers = False
            
        # Get initial observation from previous episode if reset is not called after termination or truncation
        # This may happen if the preprocessor truncates or terminates the episode under certain conditions.
        if self._new_episode and not self._reset_called:
            self._buffer[-1]['observations'] = [self._previous_eps_final_obs]
            self._new_episode = False
        
        self._buffer[-1] = self._add_to_buffer(self._buffer[-1], step_data)
            
        if (step_data['terminations'] or step_data['truncations']):
            self._buffer[-1]['seed'] = self._current_seed
            # Only check when episode is done
            if self.max_episodes_buffer is not None:
                clear_buffers = (self._episode_id % self.max_episodes_buffer == 0)
           
        if clear_buffers:
            self.clear_buffer_to_tmp_file()
        
        if step_data['terminations'] or step_data['truncations']:
            self._previous_eps_final_obs = step_data['observations']
            self._reset_called = False
            self._new_episode = True
            
            # New episode
            self._episode_id += 1
                
        if clear_buffers or step_data['terminations'] or step_data['truncations']:
            self._buffer.append({'observations': [], 'actions': [], 'terminations': [], 'truncations': [], 'rewards': []})                     
            
        return obs, rew, terminated, truncated, info                                  
                  
    def reset(self, seed=None, *args, **kwargs):
        obs, info = self.env.reset(seed=seed, *args, **kwargs)
        step_data = self._step_preprocessor(env=self, obs=obs, info=info)
        
        assert STEP_DATA_KEYS.issubset(step_data.keys())
        
        del step_data['actions']
        del step_data['rewards']
        del step_data['terminations']
        del step_data['truncations']
        
        # If reset is called before finishing last episode
        if len(self._buffer[-1]['actions']) > 0:
            # If the last episode was not term/trunc, then truncate the episode and add new episode buffer
            if not self._buffer[-1]['terminations'][-1] and not self._buffer[-1]['truncations'][-1]:
                self._buffer[-1]['truncations'][-1] = True
                self._buffer[-1]['seed'] = self._current_seed
    
            if self.max_episodes_buffer is not None and self._episode_id % self.max_episodes_buffer == 0:
                self.clear_buffer_to_tmp_file()
                
            self._buffer.append({'observations': [], 'actions': [], 'terminations': [], 'truncations': [], 'rewards': []})
            # New episode
            self._episode_id += 1
            
        self._buffer[-1] = self._add_to_buffer(self._buffer[-1], step_data)
        
        if seed is None:
            self._current_seed = str(seed)
        else:
            self._current_seed = seed
            
        self._reset_called = True
        
        return obs, info
    
    def clear_buffer_to_tmp_file(self, truncate_last_episode=False):
        def clear_buffer(buffer: dict, eps_group):
            for key, data in buffer.items():
                if isinstance(data, dict):
                    if key in eps_group:
                        eps_group_to_clear = eps_group[key]
                    else:
                        eps_group_to_clear = eps_group.create_group(key)
                    clear_buffer(data, eps_group_to_clear)      
                else:
                    # convert data to numpy
                    np_data = np.asarray(data)                        
                    assert np.all(np.logical_not(np.isnan(np_data)))
                    
                    # Check if last episode group is terminated or truncated
                    if not self._last_episode_group_term_or_trunc and key in eps_group:
                        # Append to last episode group datasets
                        if key not in STEP_DATA_KEYS:
                            # check current dataset size directly from hdf5 since
                            # additional step may not be added in a per-step/sequential basis
                            current_dataset_shape = eps_group[key].shape[0]
                        else:
                            current_dataset_shape = self._last_episode_n_steps
                            if key == 'observations':
                                current_dataset_shape +=1 # include initial observation 
                        eps_group[key].resize(current_dataset_shape + len(data), axis=0)
                        eps_group[key][-len(data):] = np_data
                    else:                      
                        if not current_episode_group_term_or_trunc:
                            # Create resizable datasets
                            eps_group.create_dataset(key, data=np_data, maxshape=(None, ) + np_data.shape[1:], chunks=True)
                        else:
                            # Dump everything to episode group
                            eps_group.create_dataset(key, data=np_data, chunks=True)
                     
        for i, eps_buff in enumerate(self._buffer):
            if len(eps_buff['actions']) == 0:
                # Make sure that there is data in the buffer and stepped 
                continue
            
            current_episode_group_term_or_trunc = eps_buff['terminations'][-1] or eps_buff['truncations'][-1]

            # Check if last episode group is terminated or truncated
            if self._last_episode_group_term_or_trunc:
                # Add new episode
                current_episode_id = self._episode_id + i + 1 - len(self._buffer)
                self._eps_group = self._tmp_f.create_group(f"episode_{current_episode_id}")
                self._eps_group.attrs['id'] = current_episode_id                  
                
            if current_episode_group_term_or_trunc:
                # Add seed to episode metadata if the current episode has finished
                # Remove seed key from episode buffer before storing datasets to file
                self._eps_group.attrs['seed'] = eps_buff.pop('seed')

            clear_buffer(eps_buff, self._eps_group)
            
            if not self._last_episode_group_term_or_trunc:
                self._last_episode_n_steps += len(eps_buff['actions'])
            else:
                self._last_episode_n_steps = len(eps_buff['actions'])
                
            if current_episode_group_term_or_trunc:
                # Compute metadata, use episode dataset in hdf5 file
                self._episode_metadata_callback(self._eps_group)                                        
            
            self._last_episode_group_term_or_trunc = current_episode_group_term_or_trunc
                        
        if not self._last_episode_group_term_or_trunc and truncate_last_episode:
            self._eps_group['truncations'][-1] = True
            self._last_episode_group_term_or_trunc = True
            self._eps_group.attrs['seed'] = self._current_seed
            
            # New episode
            self._episode_id += 1
                    
            # Compute metadata, use episode dataset in hdf5 file
            self._episode_metadata_callback(self._eps_group)
                           
        # Clear in-memory buffers
        self._buffer.clear()
    
    def save_to_disk(self, path, dataset_metadata={}):
        # Dump everything in memory buffers to tmp_dataset.hdf5 and truncate last episode       
        self.clear_buffer_to_tmp_file(truncate_last_episode=True)
        
        for key, value in dataset_metadata.items():
            self._tmp_f.attrs[key] = value
            
        self._buffer.append({'observations': [], 'actions': [], 'terminations': [], 'truncations': [], 'rewards': []})
        
        # Reset episode count
        self._episode_id = 0
        
        self._tmp_f.attrs['total_episodes'] = len(self._tmp_f.keys())
        self._tmp_f.attrs['total_steps'] = sum([episode_group.attrs['total_steps'] for episode_group in self._tmp_f.values()])
         
        # Close tmp_dataset.hdf5
        self._tmp_f.close()
        
        # Move tmp_dataset.hdf5 to specified directory 
        shutil.move(os.path.join(self._tmp_dir.name, "tmp_dataset.hdf5"), path)
            
        self._tmp_f = h5py.File(os.path.join(self._tmp_dir.name, "tmp_dataset.hdf5"), 'a', track_order=True)
            
    def close(self):
        super().close()  
         
        # Clear buffer
        self._buffer.clear()
        
        # Close tmp_dataset.hdf5
        self._tmp_f.close()
        shutil.rmtree(self._tmp_dir)
            