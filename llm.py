# app/experiments/llm.py
import os
from typing import List, Dict, Optional

import httpx
import inspect

"""
LLMClient 支持三种后端：
1) DeepSeek（优先）：如果检测到 DEEPSEEK_API_KEY
   - 优先用 LangChain: init_chat_model("deepseek:deepseek-chat")
   - 失败则走 OpenAI 兼容端点 base_url=https://api.deepseek.com model=deepseek-chat
2) OpenAI：如果检测到 OPENAI_API_KEY
3) Ollama：如果 LLM_PROVIDER=ollama（或手动设置）

优先级自动检测：DEEPSEEK > OPENAI > OLLAMA
"""

def _make_http_client():

    trust_env = os.getenv("LLM_TRUST_ENV", "0") == "1"

    # 构造 transport，按需带 retries（有的版本支持，有的不支持）
    transport_kwargs = {}
    try:
        if "retries" in inspect.signature(httpx.HTTPTransport).parameters:
            transport_kwargs["retries"] = 3
    except Exception:
        pass
    transport = httpx.HTTPTransport(**transport_kwargs)

    # 组装 client kwargs（不同版本是否支持 proxies 有差异）
    client_kwargs = dict(timeout=60.0, transport=transport, trust_env=trust_env)
    try:
        if "proxies" in inspect.signature(httpx.Client).parameters:
            client_kwargs["proxies"] = None  # 显式禁用系统代理
    except Exception:
        pass

    return httpx.Client(**client_kwargs)

class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 60):
        self.timeout = timeout

        forced = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower() or None
        has_ds = bool(os.getenv("DEEPSEEK_API_KEY"))
        has_oa = bool(os.getenv("OPENAI_API_KEY"))

        if forced in {"deepseek", "openai", "ollama"}:
            self.provider = forced
        else:
            self.provider = "deepseek" if has_ds else ("openai" if has_oa else os.getenv("LLM_PROVIDER", "openai"))

        self.model = model or os.getenv("LLM_MODEL", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", None)

        self.kind = "none"
        self.langchain_llm = None
        self.client = None
        self.http_client = _make_http_client()

        self._init_backend()

    # ---------- init ----------
    def _init_backend(self):
        prov = self.provider

        # DeepSeek
        if prov == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                prov = "openai"
            else:
                # A) LangChain deepseek（若可用）
                try:
                    from langchain.chat_models import init_chat_model
                    self.langchain_llm = init_chat_model(
                        "deepseek:deepseek-chat",
                        temperature=0,
                        api_key=api_key,
                        http_client=self.http_client,  # 关键：禁用系统代理
                    )
                    self.kind = "langchain_deepseek"
                    if not self.model:
                        self.model = "deepseek-chat"
                    return
                except Exception:
                    pass
                # B) OpenAI 兼容端点
                try:
                    from openai import OpenAI  # openai>=1.0
                    base_url = self.base_url or "https://api.deepseek.com"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        http_client=self.http_client,  # 关键：禁用系统代理
                    )
                    self.kind = "openai_compatible"
                    if not self.model:
                        self.model = "deepseek-chat"
                    return
                except Exception:
                    pass

        # OpenAI
        if prov == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI
                    if self.base_url:
                        self.client = OpenAI(api_key=api_key, base_url=self.base_url, http_client=self.http_client)
                    else:
                        self.client = OpenAI(api_key=api_key, http_client=self.http_client)
                    self.kind = "openai_compatible"
                    if not self.model:
                        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    return
                except Exception:
                    pass

        # Ollama（本地）
        if prov == "ollama":
            try:
                import requests  # noqa: F401
                self.kind = "ollama"
                if not self.base_url:
                    self.base_url = "http://localhost:11434"
                if not self.model:
                    self.model = "llama3.1"
                return
            except Exception:
                pass

        # 全部失败
        self.kind = "none"

    # ---------- chat ----------
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system"|"user"|"assistant", "content":"..."}]
        """
        if self.kind == "langchain_deepseek":
            # LangChain 的 ChatModel 推荐用 invoke([Message...])
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            lm_msgs = []
            for m in messages:
                role = (m.get("role") or "").lower()
                content = m.get("content", "")
                if not content:
                    continue
                if role == "system":
                    lm_msgs.append(SystemMessage(content=content))
                elif role == "assistant":
                    lm_msgs.append(AIMessage(content=content))
                else:
                    lm_msgs.append(HumanMessage(content=content))
            try:
                resp = self.langchain_llm.invoke(lm_msgs)
                return (getattr(resp, "content", None) or str(resp) or "").strip()
            except TypeError:
                # 兼容更老版本：退回纯文本
                text = self._messages_to_text(messages)
                try:
                    out = self.langchain_llm.invoke(text)
                    return (getattr(out, "content", None) or str(out) or "").strip()
                except Exception:
                    return self.langchain_llm.predict(text).strip()

        if self.kind == "openai_compatible":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1200,
            )
            return (resp.choices[0].message.content or "").strip()

        if self.kind == "ollama":
            import requests
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 8192, "temperature": 0}
            }
            r = requests.post(self.base_url + "/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # ollama/chat 返回 {"message":{"role":"assistant","content":"..."}}
            return (data.get("message", {}).get("content") or "").strip()

        raise RuntimeError("No LLM backend available. Set DEEPSEEK_API_KEY or OPENAI_API_KEY, or set LLM_PROVIDER=ollama and run Ollama.")

    # ---------- helper ----------
    @staticmethod
    def _messages_to_text(messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)
