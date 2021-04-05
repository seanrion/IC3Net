import make_env

env = make_env.make_env('simple_tag')
for _ in range(50):
    env.render()
env.close()