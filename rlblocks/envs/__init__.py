from gym.envs.registration import register

register(
    id='FourRoom-v0',
    entry_point='rlblocks.envs.fourroom:FourRoomEnv'
)

register(
    id='Maze-v0',
    entry_point='rlblocks.envs.maze:MazeEnv'
)

register(
    id='PaperMaze-v0',
    entry_point='rlblocks.envs.papermaze:PaperMazeEnv'
)