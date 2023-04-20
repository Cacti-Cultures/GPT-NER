# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/08/18
@desc: 这只飞很懒
"""
import os
import random
import time
from math import ceil
from typing import List

import openai
from tqdm import tqdm

from logger import get_logger

logger = get_logger(__name__)

INIT_DELAY = 1
EXPONENTIAL_BASE = 2
MAX_RETRIES = 6  # 相当于最多等待一分钟，加上前面的，就是接近两分钟


class AccessBase(object):
    # delay为类变量，不用单例是因为构造函数里有一些设置，非要用单例也行感觉没必要
    # 另外int不需要考虑线程安全问题，不用加锁
    delay = INIT_DELAY

    def __init__(self, engine, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, best_of):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of

    def _get_multiple_sample(self, prompt_list: List[str]):
        openai.api_key = os.environ["OPENAI_API_KEY"]  # 随机选择一个key
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of
        )
        results = [choice.text for choice in response.choices]
        # assert LOG_LEVEL == "INFO"
        logger.info(msg="prompt_and_result", extra={"prompt_list": prompt_list, "results": results})
        return results

    def get_multiple_sample(
            self,
            prompt_list: List[str],
            jitter: bool = True,
    ):
        """
        Retry a function with exponential backoff.
        代码借鉴自 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb 的 Example 3: Manual backoff implementation
        """
        errors: tuple = (openai.error.RateLimitError,)
        # Initialize variables
        num_retries = 0

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            used_delay = AccessBase.delay
            try:
                logger.info(f"Delay={used_delay - 1}")
                for _ in tqdm(range(ceil(used_delay - 1)), desc=f"sleep{used_delay - 1}"):
                    time.sleep(1)
                # time.sleep(used_delay - 1)  # 这里减一是为了在没有任何问题的时候不进行休眠
                results = self._get_multiple_sample(prompt_list)
                AccessBase.delay = INIT_DELAY  # 在成功之后将休眠时间置为初始值
                return results
            except errors as e:
                logger.info("重试")
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > MAX_RETRIES:
                    logger.error("重试失败")
                    raise Exception(f"Maximum number of retries ({MAX_RETRIES}) exceeded.")
                # Increment the delay
                AccessBase.delay = max(AccessBase.delay, used_delay * EXPONENTIAL_BASE * (1 + jitter * random.random()))
                # Sleep for the delay

            # Raise exceptions for any errors not specified
            except Exception as e:
                logger.info("重试")
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > MAX_RETRIES:
                    logger.error("重试失败")
                    raise Exception(f"Maximum number of retries ({MAX_RETRIES}) exceeded.")
                # Increment the delay
                AccessBase.delay = max(AccessBase.delay, used_delay * EXPONENTIAL_BASE * (1 + jitter * random.random()))
                # Sleep for the delay
