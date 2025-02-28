# coding=utf8
import os,requests,redis
from datetime import datetime

base_url = os.getenv('sr_base_url')

redis_host = os.getenv('base_redis_host')
redis_port = os.getenv('base_redis_port')
redis_pwd = os.getenv('base_redis_pwd')

redis_pool = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_pwd, db=0)
redis_conn = redis.Redis(connection_pool=redis_pool)

avatar_js = requests.get(f'{base_url}/Avatar.js')
monster1_js = requests.get(f'{base_url}/Monster_1.js')
monster2_js = requests.get(f'{base_url}/Monster_2.js')
monsterskill_js = requests.get(f'{base_url}/MonsterSkill.js')
fiction_js = requests.get(f'{base_url}/Fiction_1.js')
phantom_js = requests.get(f'{base_url}/AS.js')
chaos1_js = requests.get(f'{base_url}/Chaos_1.js')
chaos2_js = requests.get(f'{base_url}/Chaos_2.js')
curves_js = requests.get('https://homdgcat.wiki/data/LevelCurves.js')

dt_string = datetime.now().strftime('%Y%m%d%H%M%S')

redis_conn.hset('hgdcat_mhysr_datafiles', f'avatar_{dt_string}', avatar_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'monster_1_{dt_string}', monster1_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'monster_2_{dt_string}', monster2_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'monster_skill_{dt_string}', monsterskill_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'fiction_{dt_string}', fiction_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'phantom_{dt_string}', phantom_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'chaos_1_{dt_string}', chaos1_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'chaos_2_{dt_string}', chaos2_js.text)
redis_conn.hset('hgdcat_mhysr_datafiles', f'curves_{dt_string}', curves_js.text)