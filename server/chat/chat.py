from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    #负责生成聊天的响应。这个函数利用了异步迭代器，它会生成聊天的每一步响应，并根据参数 stream 来决定是一次性返回整个响应，还是以流式输出的方式逐步返回
    async def chat_iterator() -> AsyncIterable[str]:
        #使用 nonlocal 关键字声明了在外部作用域中定义的 history 和 max_tokens 变量。然后创建了一个异步迭代器回调处理器 callback，并将其添加到列表 callbacks 中。memory 变量初始化为 None，后续会根据需要赋值
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        #这部分代码负责将聊天的消息保存到数据库中。首先调用 add_message_to_db 函数将聊天消息添加到数据库，并返回消息的唯一标识符 message_id。然后创建一个对话回调处理器 conversation_callback，用于处理对话相关的回调，并将其添加到 callbacks 列表中
        # 负责保存llm response到message db
        message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="llm_chat",
                                                            query=query)
        callbacks.append(conversation_callback)
        #这段代码检查 max_tokens 是否为整数且小于等于 0，如果是，则将其设为 None。这是因为当 max_tokens 为非正数时，需要设置为 None，以表示不限制 LLM 生成 Token 的数量
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        #在这里，调用 get_ChatOpenAI 函数，根据传入的参数创建一个 LLM 模型。传入的参数包括模型名称 model_name、采样温度 temperature、最大 Token 数量 max_tokens 和回调处理器列表 callbacks
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        #这部分代码根据传入的参数 history、conversation_id 和 history_len，决定如何生成聊天的输入模板。如果 history 不为空，则优先使用前端传入的历史消息，并根据消息内容构建聊天的输入模板。如果 history 为空但传入了 conversation_id 并且 history_len 大于 0，则从数据库中获取历史消息，并根据消息构建内存对象。否则，根据传入的 prompt_name 构建聊天的输入模板
        if history: # 优先使用前端传入的历史消息
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        elif conversation_id and history_len > 0: # 前端要求从数据库取历史消息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)
        else:
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        #在这里，创建了一个 LLMChain 对象 chain，将聊天的输入模板 chat_prompt、LLM 模型 model 和内存对象 memory 传入
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)
        #这行代码创建了一个异步任务 task，用于执行聊天的逻辑。wrap_done 函数用于包装 chain.acall({"input": query})，以及 callback.done 回调函数。这个任务会在后台执行，异步处理聊天的请求
        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )
        #这里检查了 stream 参数是否为真，如果为真则表示需要流式输出
        if stream:
            #这是一个异步 for 循环，用于迭代异步迭代器 callback.aiter() 返回的结果，即聊天的响应 Token
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                #在流式输出模式下，每次生成一个 Token，通过 yield 关键字将生成的 JSON 对象发送给客户端。JSON 对象包含了 Token 的文本内容和消息的唯一标识符 message_id
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:
            answer = ""
            #再次使用异步 for 循环迭代异步迭代器 callback.aiter() 返回的结果，即聊天的响应 Token
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
