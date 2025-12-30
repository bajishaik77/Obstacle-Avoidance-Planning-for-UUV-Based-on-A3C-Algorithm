from gym.envs.registration import register

register(
    id='UUV-v0',
    entry_point='uuv_env.uuv_env:UUVEnv',
)