# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@time: 2023/01/11
@desc: 这只飞很懒
"""

import logging
import random
import threading

from friday.errors import ApiConnectionNoKeyError

logger = logging.getLogger(__name__)


class KeyManager(object):
    """
    Key管理的类
    主要是对每个key的可用性做管理，目前逻辑相对简单，只需要注意加锁保证线程安全即可
    主要功能包括提供一个可以和放回一个key，其实和queue很像
    不直接用queue的原因是：
    1. 需要提供筛选功能，即虽然所有key是一起管理的，但是不同用户的key set却不一样，而他们之间还可能有重叠，那么就需要筛选功能
    2. 需要提供排序功能，虽然优先队列可以做到，但是我们还想提供随机化功能
    这里其实应该用单例，但是项目简单，就不写单例了
    """

    def __init__(self):
        logger.info(f"初始化 KEY_TO_AVAILABILITY")
        self.mutex = threading.Lock()  # 需要线程锁保证字典的读写安全
        self.key_to_availability = dict()

    def init_key(self, key, availability):
        if key not in self.key_to_availability:
            self.key_to_availability[key] = availability

    def chose_and_occupy_a_key(self, available_key_set, request_id):
        with self.mutex:
            # 加锁后读写
            key_to_availability = {k: self.key_to_availability[k] for k in available_key_set}
            if not key_to_availability:
                logger.error("没有可用的key了")
                raise ApiConnectionNoKeyError(request_id=request_id, message="No Key")
            max_availability = max(key_to_availability.values())
            if max_availability <= 0:
                # 这里我们还没有加上【可用性为0就抛出异常】的策略，因为我们目前的管理相对严格，可用性为0其实也可以使用
                # 这部分的取舍还需要考虑
                logger.warning("key普遍被占用")
            logger.info(f"max_availability={max_availability}")
            keys = [k for k, v in key_to_availability.items() if v == max_availability]
            key = random.choice(keys)
            self.key_to_availability[key] -= 1
            return key

    def take_a_key_back(self, key):
        with self.mutex:
            # 加锁后读写
            self.key_to_availability[key] += 1


KEY_MANAGER = KeyManager()  # 偷个懒用模块实现单例了
