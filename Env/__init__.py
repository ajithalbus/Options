from gym.envs.registration import register

register(
    id='FourRooms-v0',
    entry_point='Env.envs.four_rooms:FourRooms',
    timestep_limit=100000,
)    
register(
    id='grid-v0',
    entry_point='Env.envs:GridEnv',
)

