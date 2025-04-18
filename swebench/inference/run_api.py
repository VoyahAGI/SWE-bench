#!/usr/bin/env python3

"""API推理脚本，支持OpenAI和Anthropic模型"""

import json
import os
import time
import dotenv
import traceback
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import tiktoken
import openai
from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer
from anthropic import HUMAN_PROMPT, AI_PROMPT, Anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from datasets import load_dataset, load_from_disk
from swebench.inference.make_datasets.utils import extract_diff
from argparse import ArgumentParser
import logging

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# 模型配置常量
MODEL_LIMITS = {
    "claude-instant-1": 100_000,
    "claude-2": 100_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo-16k-0613": 16_385,
    "gpt-3.5-turbo-0613": 4_097,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-4-32k-0613": 32_768,
    "gpt-4-0613": 8_192,
    "gpt-4-1106-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    'checkpoint-4070-merged-8b': 200_000,
}

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "claude-instant-1": 0.00000163,
    "claude-2": 0.00001102,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "gpt-3.5-turbo-16k-0613": 0.0000015,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-35-turbo-0613": 0.0000015,
    "gpt-35-turbo": 0.0000015,  # probably still 0613
    "gpt-4-0613": 0.00003,
    "gpt-4-32k-0613": 0.00006,
    "gpt-4-32k": 0.00006,
    "gpt-4-1106-preview": 0.00001,
    "gpt-4-0125-preview": 0.00001,
    'checkpoint-4070-merged-8b': 0.0000015,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "claude-instant-1": 0.00000551,
    "claude-2": 0.00003268,
    "claude-3-opus-20240229": 0.000075,
    "claude-3-sonnet-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000125,
    "gpt-3.5-turbo-16k-0613": 0.000002,
    "gpt-3.5-turbo-16k": 0.000002,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-35-turbo-0613": 0.000002,
    "gpt-35-turbo": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4-32k-0613": 0.00012,
    "gpt-4-32k": 0.00012,
    "gpt-4-1106-preview": 0.00003,
    "gpt-4-0125-preview": 0.00003,
    'checkpoint-4070-merged-8b': 0.000002,
}

# used for azure
ENGINES = {
    "gpt-3.5-turbo-16k-0613": "gpt-35-turbo-16k",
    "gpt-4-0613": "gpt-4",
    "gpt-4-32k-0613": "gpt-4-32k",
}

def calc_cost(model_name, input_tokens, output_tokens):
    """计算API调用成本"""
    cost = (
        MODEL_COST_PER_INPUT.get(model_name, 0) * input_tokens
        + MODEL_COST_PER_OUTPUT.get(model_name, 0) * output_tokens
    )
    logger.info(f"Tokens: {input_tokens} in, {output_tokens} out | Cost: ${cost:.5f}")
    return cost

@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_chat(client, model_name, messages, temperature=0.7, top_p=0.95, **kwargs):
    """通用OpenAI风格API调用"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # 计算成本
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calc_cost(model_name, input_tokens, output_tokens)
        
        return response, cost
    
    except openai.APIConnectionError as e:
        logger.error(f"连接错误: {e.__cause__}")
        raise
    except openai.RateLimitError as e:
        logger.warning("速率限制，等待重试...")
        time.sleep(30)
        raise
    except openai.APIStatusError as e:
        logger.error(f"API错误: {e.status_code} {e.message}")
        raise

def create_openai_client(model_args):
    """创建OpenAI客户端实例"""
    use_azure = model_args.pop("use_azure", False)
    api_base = model_args.pop("api_base", None)
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")  # 虚拟密钥兼容本地API
    
    if use_azure:
        return AzureOpenAI(
            api_key=api_key,
            api_version=model_args.pop("api_version", "2023-05-15"),
            azure_endpoint=api_base or "https://pnlpopenai3.openai.azure.com/"
        )
    else:
        return OpenAI(
            api_key=api_key,
            base_url=api_base
        )

def openai_inference(
    test_dataset,
    model_name,
    output_file,
    model_args,
    existing_ids,
    max_cost,
):
    """OpenAI风格API推理主流程"""
    # 初始化客户端
    client = create_openai_client(model_args)
    
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/voyah/Workspace/zhl/data/tokenizer", 
            use_fast=True
        )
    except Exception as e:
        logger.warning(f"无法加载分词器: {e}, 使用tiktoken")
        tokenizer = tiktoken.encoding_for_model(model_name)
    
    # 数据集过滤
    test_dataset = test_dataset.filter(
        lambda x: len(tokenizer.encode(x["text"])) <= MODEL_LIMITS[model_name],
        desc="过滤超长上下文",
        load_from_cache_file=False,
    )
    
    # 推理参数
    temperature = model_args.pop("temperature", 0.7)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    logger.info(f"推理参数: temperature={temperature}, top_p={top_p}")

    total_cost = 0
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"推理进度 {model_name}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            
            # 构造消息
            system_msg, _, user_msg = datum["text"].partition("\n")
            messages = [
                {"role": "system", "content": system_msg.strip()},
                {"role": "user", "content": user_msg.strip()},
            ]
            
            try:
                response, cost = call_chat(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    **model_args
                )
            except Exception as e:
                logger.error(f"实例 {instance_id} 处理失败: {str(e)}")
                continue
            
            total_cost += cost
            completion = response.choices[0].message.content
            
            # 保存结果
            output = {
                "instance_id": instance_id,
                "model": model_name,
                "full_output": completion,
                "model_patch": extract_diff(completion),
                "cost": cost,
                "timestamp": time.time(),
            }
            f.write(json.dumps(output) + "\n")
            f.flush()
            
            logger.info(f"累计成本: ${total_cost:.5f}")
            if max_cost and total_cost >= max_cost:
                logger.warning(f"达到最大成本限制 ${max_cost}, 停止推理")
                break

def create_anthropic_client(model_args):
    """创建Anthropic客户端实例"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY环境变量未设置")
    
    return Anthropic(api_key=api_key)

@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(5))
def call_anthropic_api(client, model_name, prompt, temperature=0.2, top_p=0.95, **kwargs):
    """统一Anthropic API调用"""
    try:
        # Claude 3使用messages接口
        if "claude-3" in model_name:
            system_prompt, _, user_message = prompt.partition("\n")
            messages = [{"role": "user", "content": user_message.strip()}]
            
            response = client.messages.create(
                model=model_name,
                messages=messages,
                system=system_prompt.strip(),
                max_tokens=4096,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            content = response.content[0].text
            
        else:  # Claude 2及以下版本
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens_to_sample=4096,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            input_tokens = client.count_tokens(prompt)
            output_tokens = client.count_tokens(response.completion)
            content = response.completion
        
        cost = calc_cost(model_name, input_tokens, output_tokens)
        return content, cost
    
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        if hasattr(e, "status_code"):
            if e.status_code == 429:  # 速率限制
                logger.warning("速率限制，等待10秒后重试...")
                time.sleep(10)
            elif 500 <= e.status_code < 600:  # 服务器错误
                logger.warning(f"服务器错误 ({e.status_code}), 等待30秒...")
                time.sleep(30)
        raise

def anthropic_inference(
    test_dataset,
    model_name,
    output_file,
    model_args,
    existing_ids,
    max_cost,
):
    """Anthropic模型推理主流程"""
    # 初始化客户端
    client = create_anthropic_client(model_args)
    
    # 过滤数据集
    test_dataset = test_dataset.map(
        lambda x: {"token_count": client.count_tokens(x["text"])},
        desc="计算Token数量",
        load_from_cache_file=False,
    ).filter(
        lambda x: x["token_count"] <= MODEL_LIMITS[model_name],
        desc="过滤超长上下文",
        load_from_cache_file=False,
    )
    
    # 推理参数
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    logger.info(f"推理参数: temperature={temperature}, top_p={top_p}")

    total_cost = 0
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"推理进度 {model_name}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            
            # 构造Prompt
            if "claude-3" in model_name:
                prompt = datum["text"]
            else:
                prompt = f"{HUMAN_PROMPT} {datum['text']}\n\n{AI_PROMPT}"
            
            try:
                completion, cost = call_anthropic_api(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    **model_args
                )
            except Exception as e:
                logger.error(f"实例 {instance_id} 处理失败: {str(e)}")
                continue
            
            total_cost += cost
            output = {
                "instance_id": instance_id,
                "model": model_name,
                "full_output": completion,
                "model_patch": extract_diff(completion),
                "cost": cost,
                "timestamp": time.time(),
            }
            f.write(json.dumps(output) + "\n")
            f.flush()
            
            logger.info(f"累计成本: ${total_cost:.5f}")
            if max_cost and total_cost >= max_cost:
                logger.warning(f"达到最大成本限制 ${max_cost}, 停止推理")
                break

def parse_model_args(args_str):
    """解析模型参数字符串"""
    kwargs = {}
    if not args_str:
        return kwargs
    
    for pair in args_str.split(","):
        key, value = pair.split("=", 1)
        try:
            # 自动类型转换
            if value.lower() == "true":
                kwargs[key] = True
            elif value.lower() == "false":
                kwargs[key] = False
            elif value.isdigit():
                kwargs[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                kwargs[key] = float(value)
            else:
                kwargs[key] = value
        except:
            kwargs[key] = value
    return kwargs

def main(args):
    """主流程"""
    # 输出文件处理
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集加载
    if Path(args.dataset_name_or_path).exists():
        dataset = load_from_disk(args.dataset_name_or_path)
    else:
        dataset = load_dataset(args.dataset_name_or_path)
    
    # 过滤已处理实例
    existing_ids = set()
    output_file = output_dir / f"{args.model_name_or_path}_results.jsonl"
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data["instance_id"])
    
    # 排序处理
    dataset = dataset[args.split]
    indices = np.argsort([len(x) for x in dataset["text"]])
    dataset = dataset.select(indices)
    
    # 分片处理
    if args.shard_id is not None and args.num_shards:
        dataset = dataset.shard(args.num_shards, args.shard_id)
    
    # 选择推理引擎
    if "claude" in args.model_name_or_path.lower():
        anthropic_inference(...)
    else:
        openai_inference(
            test_dataset=dataset,
            model_name=args.model_name_or_path,
            output_file=output_file,
            model_args=parse_model_args(args.model_args),
            existing_ids=existing_ids,
            max_cost=args.max_cost,
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name of API model. Update MODEL* constants in this file to add new models.",
        choices=sorted(list(MODEL_LIMITS.keys())),
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Maximum cost to spend on inference.",
    )
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"主流程异常: {str(e)}")
        traceback.print_exc()
        exit(1)