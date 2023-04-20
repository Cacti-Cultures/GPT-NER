# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@time: 2022/08/18
@desc: 提供和openai的交互接口，在openai提供的接口的基础上增加一些常用函数
"""
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict

import openai
from math import ceil
from more_itertools import chunked
from prometheus_client import Gauge, Counter

from friday.errors import ApiConnectionError, ApiConnectionTimeoutError, ApiConnectionUnknownError
from friday.key_manager import KEY_MANAGER

logger = logging.getLogger(__name__)
KEY_TO_ALGORITHM_DELAY_GAUGE = dict()
KEY_TO_ALGORITHM_RETRY_COUNT = dict()
KEY_TO_DELAY = dict()
INIT_DELAY = 1
EXPONENTIAL_BASE = 2


class ApiConnector(object):
    """
    通过KEY_MANAGER获取key，并对大量prompt进行请求，封装了一些常用的函数，避免反复写多线程
    目前没有使用async，因为我们的io没有密集到那个程度
    """

    def __init__(self, api_key_list, max_sleep_time, max_retries):
        logger.info(f"初始化ApiConnector, key-size={len(api_key_list)}")
        self.api_key_list = api_key_list
        self.max_sleep_time = max_sleep_time
        self.max_retries = max_retries
        for key in self.api_key_list:
            KEY_MANAGER.init_key(key, 1)  # 初始化key对应的available count，如果已经存在了就不会初始化，这个逻辑在KeyManager里
            if key not in KEY_TO_DELAY:
                KEY_TO_DELAY[key] = INIT_DELAY
            # 下面的代码是针对普罗米修斯的，没有普罗米修斯的可以忽略
            if key not in KEY_TO_ALGORITHM_DELAY_GAUGE:
                key_name = key[-4:]
                KEY_TO_ALGORITHM_DELAY_GAUGE[key] = Gauge(
                    f"friday_algorithm_delay_{key_name}",
                    f"friday_algorithm_delay_{key_name}.value"
                )
                KEY_TO_ALGORITHM_RETRY_COUNT[key] = Counter(
                    f"friday_algorithm_retry_{key_name}",
                    f"friday_algorithm_retry_{key_name}.value"
                )

    @staticmethod
    def call_api(*, api_key: str, prompt_list: List[str], inference_parameters: Dict) -> List[str]:
        """
        这个函数就是直接调用openai了，只是加了一点判断和log
        """
        assert set(inference_parameters.keys()) == {
            "request_timeout", "engine", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
            "best_of",
        }
        # 不要使用timeout这个参数
        response = openai.Completion.create(
            api_key=api_key,
            prompt=prompt_list,
            **inference_parameters
        )
        results = [choice.text for choice in response.choices]
        # assert LOG_LEVEL == "INFO"
        logger.info(
            msg=f"prompt_and_result",
            extra={"parameters": inference_parameters, "prompt_list": prompt_list, "results": results}
        )
        return results

    def get_available_key_count(self) -> int:
        """
        这里获取可用的key的数量，用于后面决定并发的worker数量
        目前逻辑很简单，只是简单返回长度，后面可能可以根据不同的key的类型，给出不同的加权
        """
        return len(self.api_key_list)

    def chunking(self, prompt_list: List[str], inference_parameters) -> List[List[str]]:
        """
        将输入的长列表分batch
        其中最重要的chunk size决定于两个因素：
        1. 不能超过token rate limit，这个大约摸索了一个公式出来，但是目前还有点瑕疵
        2. 当前可用key的数量
        这二者各自决定了一个chunk size，我们取最小值，也就是尽量分的小一些，避免超过limit
        """
        available_key_count = self.get_available_key_count()
        logger.info(f"可用key数量为{available_key_count}")
        chunk_size = min(8, min(
            150000 // (inference_parameters["max_tokens"] * 20),
            ceil(len(prompt_list) / available_key_count)
        ))

        assert chunk_size > 0, inference_parameters
        return list(chunked(prompt_list, chunk_size))

    def get_multiple_sample_list_of_list(self, prompt_list_list: List[List[str]], inference_parameters, request_id):
        """
        有的时候我们每一个instance有多个prompt，而且他们的数量可能不一致，为了避免大家自己写对齐的逻辑，这里提供一个简单函数来对其
        """
        squashed_prompt_list = []
        origin_to_squash = []
        for prompt_list in prompt_list_list:
            current_length = len(squashed_prompt_list)
            squashed_prompt_list.extend(prompt_list)
            origin_to_squash.append(range(current_length, current_length + len(prompt_list)))
        squashed_results = self.get_multiple_sample_in_parallel(
            prompt_list=squashed_prompt_list, inference_parameters=inference_parameters, request_id=request_id
        )
        origin_results = []
        for idx_list in origin_to_squash:
            origin_results.append([squashed_results[idx] for idx in idx_list])
        for prompt_list, results in zip(prompt_list_list, origin_results):
            assert len(prompt_list) == len(results)
        return origin_results

    def get_multiple_sample_in_parallel(self, *, prompt_list: List[str], inference_parameters, request_id) -> List[str]:
        """
        并发执行，因为开线程的代价不高，所以没有把线程池提出来
        """
        prompt_list_chunks = self.chunking(prompt_list, inference_parameters)
        logger.info(
            f"并发请求 {request_id}，数据共{len(prompt_list)}，chunk数={len(prompt_list_chunks)}"
        )

        with ThreadPoolExecutor(max_workers=len(prompt_list_chunks)) as exe:
            results_for_each_chunk = list(
                exe.map(
                    self.get_multiple_sample_with_error_handling,
                    prompt_list_chunks,
                    [inference_parameters] * len(prompt_list_chunks),
                    [request_id] * len(prompt_list_chunks)
                )
            )
        return [result for results in results_for_each_chunk for result in results]

    def get_multiple_sample_with_error_handling(self, prompt_list: List[str], inference_parameters: Dict,
                                                request_id: str) -> List[str]:
        """
        这个函数很简单，就是对get_multiple_sample的封装。
        需要这个函数原因是：我们的并发不希望因为某一个请求失败而全部失败（损失太大了）；同时我们还希望能够给错误结果一个同类型返回值。
        而把这个单独拎出来的原因，是想保持get_multiple_sample函数的逻辑清晰一些，毕竟那个函数已经挺长的了
        """
        try:
            return self.get_multiple_sample(prompt_list=prompt_list, inference_parameters=inference_parameters,
                                            request_id=request_id)
        except ApiConnectionError as e:
            logger.error(e, exc_info=True)
            return [f"FRIDAY-ERROR-{e.error_type}" for _ in prompt_list]

    def get_multiple_sample(self, prompt_list: List[str], inference_parameters: Dict, request_id: str) -> List[str]:
        """
        带重试和各类错误处理的主函数
        其中exponential backoff的代码借鉴自
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
        的 Example 3: Manual backoff implementation
        这个函数的最主要逻辑是：
        请求openai，遇到错误的话根据错误信息判断是重试（可能需要休眠）还是直接返回
        """
        assert set(inference_parameters.keys()) == {
            "request_timeout", "engine", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
            "best_of",
        }, inference_parameters.keys()
        # Initialize variables
        num_retries = 0
        total_delay = 0
        while total_delay < self.max_sleep_time and num_retries < self.max_retries:
            logger.info(f"key_pool_size={len(self.api_key_list)}")
            api_key = KEY_MANAGER.chose_and_occupy_a_key(self.api_key_list, request_id)  # 随机选择一个key
            logger.info(f"Use key {api_key}")
            used_delay = KEY_TO_DELAY[api_key]
            try:
                logger.info(f"Delay={used_delay - 1}")
                KEY_TO_ALGORITHM_DELAY_GAUGE[api_key].set(used_delay - 1)  # 记录当前sleep的时间
                # 把休眠放到前面而不是后面，目的是优先完成用户请求，下次遇到这个key再说
                # 这里其实可以优化，因为下次遇到这个key已经是过了一会儿了，但是这个逻辑还没加，无伤大雅
                logger.info(f"sleep {used_delay - 1}")
                time.sleep(used_delay - 1)
                total_delay += used_delay
                results = self.call_api(api_key=api_key, prompt_list=prompt_list,
                                        inference_parameters=inference_parameters)
                KEY_TO_DELAY[api_key] = INIT_DELAY  # 在成功之后将休眠时间置为初始值
                return results
            except openai.error.RateLimitError as e:
                # Rate limit分两种情况：
                # credit不足和超过limit，前者代表这个key不能再使用了，后者代表需要休眠
                logger.error(e)
                if e.json_body["error"]["type"] == "insufficient_quota":
                    logger.error("insufficient_quota")
                    logger.info(f"pop key {api_key}")
                    if api_key in self.api_key_list:
                        self.api_key_list.remove(api_key)
                    logger.info(f"剩余key数量={len(self.api_key_list)}")
                    logger.info("保存余额不足的api_key")
                    with open("./api_key_invalid_record.jsonl", "a") as output_file:
                        content = {
                            "api_key": api_key,
                            "datetime": f"{datetime.now():%Y-%m-%d %H-%M-%S}"
                        }
                        output_file.write(json.dumps(content) + "\n")
                else:
                    logger.info(f"重试")
                    logger.info(
                        f"触发重试: key={api_key}, prompt_list={prompt_list}",
                        extra={"prompt_list": prompt_list, "key": api_key}
                    )
                    if used_delay == 1:
                        logger.info(
                            f"第一次触发重试: prompt_list={prompt_list}",
                            extra={"prompt_list": prompt_list, "key": api_key}
                        )
                    # Increment the delay
                    # KEY_TO_DELAY[api_key] = min(
                    #     max(
                    #         KEY_TO_DELAY[api_key],
                    #         used_delay * EXPONENTIAL_BASE * (1 + random.random())
                    #     ),
                    #     self.max_sleep_time
                    # )
                    KEY_TO_DELAY[api_key] += 1
            except openai.error.ServiceUnavailableError as e:
                # 远端服务不可用，这个其实比较恶心，因为有时会过比较长的时间才抛出，这也是我们增加timeout的原因
                logger.error(f"远端服务不可用 ServiceUnavailableError, key={api_key}")
                total_delay += 3
                num_retries += 1
                KEY_TO_ALGORITHM_RETRY_COUNT[api_key].inc(1)
                time.sleep(3)
            except openai.error.Timeout as e:
                # 这个代表请求超过了用户设置的timeout，直接raise error即可
                with open("./timeout.jsonl", "a") as output_file:
                    content = {
                        "api_key": api_key,
                        "datetime": f"{datetime.now():%Y-%m-%d %H-%M-%S}"
                    }
                    output_file.write(json.dumps(content) + "\n")
                raise ApiConnectionTimeoutError(request_id=request_id, message="超时")
            except Exception as e:
                # 其他异常，比较罕见
                logger.error(f"其他错误")
                logger.error(e)
                logger.info(f"触发其他错误: prompt_list={prompt_list}", extra={"prompt_list": prompt_list})
                raise ApiConnectionUnknownError(request_id=request_id, message="其他错误")
            finally:
                # 这里我们强制休眠了两秒，是为了稳定性增加的强制措施，两秒后再将api_key回收，也就是说，在此之前是被占用状态
                # 在并发量大的时候，这两秒可以忽略，之前测试感觉影响不大
                time.sleep(2)
                KEY_MANAGER.take_a_key_back(api_key)
        raise ApiConnectionUnknownError(
            request_id=request_id,
            message=f"可能总延时过大，可能重试次数过多，总之直接退出，total_delay={total_delay}, n_tries={num_retries}"
        )
