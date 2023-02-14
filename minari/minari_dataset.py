import os
import h5py
import warnings
import gymnasium as gym
from typing import Optional
from gymnasium.envs.registration import EnvSpec
from minari.utils.data_collector import DataCollectorV0
from minari.storage.datasets_root_dir import get_dataset_path


class MinariDataset:
    def __init__(self,
        data_path: str
    ):
        """The `id` parameter corresponds to the name of the dataset, with the syntax as follows:
    `(namespace)/(env_name)-v(version)` where `namespace` is optional. 
        """
               
        self._data_path = data_path
        self._extra_data_id = 0
        with h5py.File(self._data_path, 'r') as f:
            self._flatten_observations = f.attrs['flatten_observation']
            self._flatten_actions = f.attrs['flatten_action']
            self._env_spec = EnvSpec.from_json(f.attrs['env_spec'])

            self._total_episodes = f.attrs['total_episodes']
            self._total_steps = f.attrs['total_steps']
            
            self._dataset_name = f.attrs['dataset_name']
            self._combined_datasets = f.attrs.get('combined_datasets')
            
            env = gym.make(self._env_spec)
            
            self._observation_space = env.observation_space
            self._action_space = env.action_space

            env.close()
            
    def recover_environment(self):
        return gym.make(self._env_spec)
    
    @property
    def flatten_observations(self) -> bool:
        return self._flatten_observations
    
    @property
    def flatten_actions(self)-> bool:
        return self._flatten_actions
    
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def data_path(self):
        return self._data_path
    
    @property
    def total_steps(self):
        return self._total_steps
    
    @property
    def total_episodes(self):
        return self._total_episodes
    
    @property
    def combined_datasets(self):
        if self._combined_datasets is None:
            return []
        else:
            return self._combined_datasets
    
    @property
    def name(self):
        return self._dataset_name
    
    def update_dataset_from_collector_env(self, collector_env: DataCollectorV0):
        # check that collector env has the same characteristics as self._env_spec
        
        new_data_file_path = os.path.join(os.path.split(self.data_path)[0], f"additional_data_{self._extra_data_id}.hdf5")
        
        collector_env.save_to_disk(path=new_data_file_path)
        
        with h5py.File(new_data_file_path, 'r', track_order=True) as new_data_file:
            group_paths = [group.name for group in new_data_file.values()]
            new_data_total_episodes = new_data_file.attrs['total_episodes']
            new_data_total_steps = new_data_file.attrs['total_steps']
            
        with h5py.File(self.data_path, 'a', track_order=True) as file:
            last_episode_id = file.attrs['total_episodes']
            for i, eps_group_path in enumerate(group_paths):
                file[f"episode_{last_episode_id + i}"] = h5py.ExternalLink(f"additional_data_{self._extra_data_id}.hdf5", eps_group_path)
                file[f"episode_{last_episode_id + i}"].attrs.modify('id', last_episode_id + i)
            
            # Update metadata of minari dataset
            file.attrs.modify('total_episodes', last_episode_id + new_data_total_episodes)
            file.attrs.modify('total_steps', file.attrs['total_steps'] + new_data_total_steps)
        self._extra_data_id += 1
               
    def update_dataset_from_buffers(self,
                                    observations,
        actions,
        rewards,
        terminations,
        truncations,
        seeds=None
        ):
        
        # NoneType warning for seeds
        pass

def combine_datasets(datasets_to_combine: list[MinariDataset], new_dataset_name: str):
    new_dataset_path = get_dataset_path(new_dataset_name)
    
    # Check if dataset already exists
    if not os.path.exists(new_dataset_path):
        new_dataset_path = os.path.join(new_dataset_path, "data")
        os.makedirs(new_dataset_path)
        new_data_path = os.path.join(new_dataset_path, "main_data.hdf5")
    else:
        raise ValueError(f'A Minari dataset with ID {new_dataset_name} already exists and it cannot be overriden. Please use a different dataset name or version.')
    
    with h5py.File(new_data_path, 'a', track_order=True) as combined_data_file:
        combined_data_file.attrs['total_episodes'] = 0
        combined_data_file.attrs['total_steps'] = 0
        combined_data_file.attrs['dataset_name'] = new_dataset_name
        
        combined_data_file.attrs['combined_datasets'] = [dataset.name for dataset in datasets_to_combine]
        
        for dataset in datasets_to_combine:
            if not isinstance(dataset, MinariDataset):
                raise ValueError(f"The dataset {dataset} is not of type MinariDataset.")
            
            with h5py.File(dataset.data_path, 'r', track_order=True) as data_file:
                group_paths = [group.name for group in data_file.values()]
                
                if combined_data_file.attrs.get('env_spec') is None:
                    combined_data_file.attrs['env_spec'] = data_file.attrs['env_spec']
                else:
                    if combined_data_file.attrs['env_spec'] != data_file.attrs['env_spec']:
                        raise ValueError ("The datasets to be combined have different values for `env_spec` attribute.")
                
            if combined_data_file.attrs.get('flatten_action') is None:
                combined_data_file.attrs['flatten_action'] = dataset.flatten_actions
            else:
                if combined_data_file.attrs['flatten_action'] != dataset.flatten_actions:
                    raise ValueError ("The datasets to be combined have different values for `flatten_action` attribute.")
                
            if combined_data_file.attrs.get('flatten_observation') is None:
                combined_data_file.attrs['flatten_observation'] = dataset.flatten_observations
            else:
                if combined_data_file.attrs['flatten_observation'] != dataset.flatten_observations:
                    raise ValueError ("The datasets to be combined have different values for `flatten_observation` attribute.")
            
            last_episode_id = combined_data_file.attrs['total_episodes']
            
            for i, eps_group_path in enumerate(group_paths):
                combined_data_file[f"episode_{last_episode_id + i}"] = h5py.ExternalLink(dataset.data_path, eps_group_path)
                combined_data_file[f"episode_{last_episode_id + i}"].attrs.modify('id', last_episode_id + i)
            
            # Update metadata of minari dataset
            combined_data_file.attrs.modify('total_episodes', last_episode_id + dataset.total_episodes)
            combined_data_file.attrs.modify('total_steps', combined_data_file.attrs['total_steps'] + dataset.total_steps)
    
    return MinariDataset(new_data_path)

def create_dataset_from_buffers(dataset_name: str,
        algorithm_name: str,
        environment,
        code_permalink,
        author,
        author_email,
        observations,
        actions,
        rewards,
        terminations,
        truncations,
        ):
    
     # NoneType warnings
    if code_permalink is None:
        warnings.warn("`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.", UserWarning)
    if author is None:
        warnings.warn("`author` is set to None. For longevity purposes it is highly recommended to provide an author name.", UserWarning)
    if author_email is None:
        warnings.warn("`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.", UserWarning)

    
def create_dataset_from_collector_env(dataset_name,
        collector_env: DataCollectorV0,
        algorithm_name:Optional[str]=None,
        author:Optional[str]=None,
        author_email:Optional[str]=None,
        code_permalink:Optional[str]=None,):
    
    # NoneType warnings
    if code_permalink is None:
        warnings.warn("`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.", UserWarning)
    if author is None:
        warnings.warn("`author` is set to None. For longevity purposes it is highly recommended to provide an author name.", UserWarning)
    if author_email is None:
        warnings.warn("`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.", UserWarning)
        
    dataset_path = os.path.join(collector_env.datasets_path, dataset_name)
    
    # Check if dataset already exists
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(dataset_path, "data")
        os.makedirs(dataset_path)
        data_path = os.path.join(dataset_path, "main_data.hdf5")
        collector_env.save_to_disk(data_path, dataset_metadata={"dataset_name": str(dataset_name),
                                                                "algorithm_name": str(algorithm_name),
                                                                "author": str(author),
                                                                "author_email": str(author_email),
                                                                "code_permalink": str(code_permalink),
                                                                    })
        return MinariDataset(data_path)    
    else:
        raise ValueError(f'A Minari dataset with ID {dataset_name} already exists and it cannot be overriden. Please use a different dataset name or version.')